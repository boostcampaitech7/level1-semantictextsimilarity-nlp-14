import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer 클래스
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch 기반 훈련
            self.len_epoch = len(self.data_loader)
        else:
            # iteration 기반 훈련
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        에포크 당 훈련 로직

        :param epoch: Integer, 현재 훈련 에포크.
        :return: 평균 손실 및 메트릭을 포함하는 로그.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, batch in enumerate(self.data_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(input_ids, attention_mask)
            loss = self.criterion(output, labels)

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, labels))

            if batch_idx % self.log_step == 0:
                self.logger.debug('훈련 에포크: {} {} 손실: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            # 에포크 종료 조건 수정
            if batch_idx >= self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _valid_epoch(self, epoch):
        """
        에포크 후 검증

        :param epoch: Integer, 현재 훈련 에포크.
        :return: 검증에 대한 정보를 포함하는 로그.
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                output = self.model(input_ids, attention_mask)
                loss = self.criterion(output, labels)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, labels))

        # 모델 파라미터의 히스토그램을 텐서보드에 추가
        #for name, p in self.model.named_parameters():
            #self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

