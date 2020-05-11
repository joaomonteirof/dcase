import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from tqdm import tqdm
from models.losses import LabelSmoothingLoss

class TrainLoop(object):

	def __init__(self, model, optimizer, train_loader, valid_loader, max_gnorm, label_smoothing, verbose=-1, cp_name=None, save_cp=False, checkpoint_path=None, checkpoint_epoch=None, cuda=True, logger=None):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, cp_name) if cp_name else os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.model = model
		self.optimizer = optimizer
		self.max_gnorm = max_gnorm
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.total_iters = 0
		self.cur_epoch = 0
		self.verbose = verbose
		self.save_cp = save_cp
		self.device = next(self.model.parameters()).device
		self.logger = logger
		self.history = {'train_loss': [], 'train_loss_batch': []}
		self.disc_label_smoothing = label_smoothing
		self.best_er = np.inf

		if label_smoothing>0.0:
			self.ce_criterion = LabelSmoothingLoss(label_smoothing, lbl_set_size=self.model.n_classes)
		else:
			self.ce_criterion = torch.nn.CrossEntropyLoss()

		if self.valid_loader is not None:
			self.history['ER'] = []

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1, eval_every=1000):

		while (self.cur_epoch < n_epochs):

			self.cur_epoch += 1

			np.random.seed()

			if self.verbose>0:
				print(' ')
				print('Epoch {}/{}'.format(self.cur_epoch, n_epochs))
				train_iter = tqdm(enumerate(self.train_loader))
			else:
				train_iter = enumerate(self.train_loader)

			self.save_epoch_cp = False
			train_loss_epoch=0.0
			for t, batch in train_iter:
				train_loss = self.train_step(batch)
				self.history['train_loss_batch'].append(train_loss)
				train_loss_epoch+=train_loss

				self.total_iters += 1

				if self.logger:
					self.logger.add_scalar('Train/Train Loss', train_loss, self.total_iters)
					self.logger.add_scalar('Info/LR', self.optimizer.optimizer.param_groups[0]['lr'], self.total_iters)

				if self.total_iters % eval_every == 0:
					self.evaluate()
					if self.save_cp and ( self.history['ER'][-1] < np.min([np.inf]+self.history['ER'][:-1]) ):
							self.checkpointing()
							self.save_epoch_cp = True

				self.history['train_loss'].append(train_loss_epoch/(t+1))

				if self.verbose>0:
					print(' ')
					print('Total train loss: {:0.4f}'.format(self.history['train_loss'][-1]))
					print('Current LR: {}'.format(self.optimizer.optimizer.param_groups[0]['lr']))
					print(' ')

			if self.save_cp and self.cur_epoch % save_every == 0 and not self.save_epoch_cp:
					self.checkpointing()

		if self.verbose>0:
			print('Training done!')

		if self.valid_loader is not None:
			if self.verbose>0:
				print('Best error rate and corresponding epoch and iteration: {:0.4f}, {}, {}'.format(np.min(self.history['ER']), self.best_er_epoch, self.best_er_iteration))

			return np.min(self.history['ER'])
		else:
			return [np.min(self.history['train_loss'])]

	def train_step(self, batch):

		self.model.train()
		self.optimizer.zero_grad()

		x, y = batch

		x = x.to(self.device, non_blocking=True)
		y = y.to(self.device, non_blocking=True)

		if random.random() > 0.5:
			x += torch.randn_like(x)*random.choice([1e-4, 1e-5])

		out = self.model.forward(x)

		loss = self.ce_criterion(out, y)

		loss.backward()

		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gnorm)

		self.optimizer.step()

		if self.logger:
			self.logger.add_scalar('Info/Grad_norm', grad_norm, self.total_iters)

		return loss.item()

	def valid(self, batch):

		self.model.eval()

		with torch.no_grad():

			x, y = batch

			x = x.to(self.device, non_blocking=True)
			y = y.to(self.device, non_blocking=True)

			out = self.model.forward(x)

			_, pred = output.topk(1, 1, True, True)
			pred = pred.t()
			correct = pred.eq(target.view(1, -1).expand_as(pred))

			correct = correct[:1].view(-1).float().sum(0, keepdim=True)

		return correct

	def evaluate(self):

		if self.verbose>0:
			print('\nIteration - Epoch: {} - {}'.format(self.total_iters, self.cur_epoch))

		total_correct, total = 0, 0

		for t, batch in enumerate(self.valid_loader):
			correct = self.valid(batch)
			total_correct += correct
			total += batch.size(0)

		self.history['ER'].append(1.-total_correct/total)

		if self.history['ER'][-1]<self.best_er:
			self.best_er = self.history['ER'][-1]
			self.best_er_epoch = self.cur_epoch
			self.best_er_iteration = self.total_iters

		if self.logger:
			self.logger.add_scalar('Valid/ER', self.history['ER'][-1], self.total_iters)
			self.logger.add_scalar('Valid/Best ER', np.min(self.history['ER']), self.total_iters)

		if self.verbose>0:
			print(' ')
			print('Current ER, best ER, and epoch - iteration: {:0.4f}, {:0.4f}, {}, {}'.format(self.history['ER'][-1], np.min(self.history['ER']), self.best_er_epoch, self.best_er_iteration))

	def checkpointing(self):

		# Checkpointing
		if self.verbose>0:
			print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'n_classes': self.model.n_classes,
		'optimizer_state': self.optimizer.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch}
		try:
			torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))
		except:
			torch.save(ckpt, self.save_epoch_fmt)

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt, map_location = lambda storage, loc: storage)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			if self.cuda_mode:
				self.model = self.model.cuda(self.device)

		else:
			print('No checkpoint found at: {}'.format(ckpt))