import argparse
import math

import mlconfig
import torch
from torch import distributed, nn

from efficientnet.utils import distributed_is_initialized
import torch.optim.lr_scheduler as lr_scheduler

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', type=str, required=True)
	parser.add_argument('--resume', type=str, default=None)
	parser.add_argument('--no-cuda', action='store_true')
	parser.add_argument('--data-parallel', action='store_true')
	parser.add_argument('--log-dir', type=str, default="logs")

	# distributed
	parser.add_argument('--backend', type=str, default='nccl')
	parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:23456')
	parser.add_argument('--world-size', type=int, default=1)
	parser.add_argument('--rank', type=int, default=0)

	return parser.parse_args()


def init_process(backend, init_method, world_size, rank):
	distributed.init_process_group(
		backend=backend,
		init_method=init_method,
		world_size=world_size,
		rank=rank,
	)


def main():
	torch.backends.cudnn.benchmark = True

	args = parse_args()
	config = mlconfig.load(args.config)
	print(config)

	
	if args.world_size > 1:
		init_process(args.backend, args.init_method, args.world_size, args.rank)

	device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

	model = config.model()
	if distributed_is_initialized():
		model.to(device)
		model = nn.parallel.DistributedDataParallel(model)
	else:
		if args.data_parallel:
			num_gpus = torch.cuda.device_count()
			model = nn.DataParallel(model, device_ids=[i for i in range(num_gpus)])
		model.to(device)

	lr0 = 0.0001 * config.dataset.batch_size
	lf = lambda x: ((1 + math.cos(x * math.pi / config.trainer.num_epochs)) / 2) * (1 - 0.01) + 0.01  # cosine

	# optimizer = config.optimizer(model.parameters(), lr=lr0)
	optimizer = config.optimizer(model.parameters(), lr=lr0, betas=(0.937, 0.999))
	scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

	train_loader = config.dataset(train=True)
	valid_loader = config.dataset(train=False)
	
	
	trainer = config.trainer(model, optimizer, train_loader, valid_loader, scheduler, device)

	if args.resume is not None:
		trainer.resume(args.resume)

	trainer.fit()
	

if __name__ == "__main__":
	main()
