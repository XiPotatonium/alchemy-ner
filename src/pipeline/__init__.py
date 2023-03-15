import copy
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional
from loguru import logger
import torch
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from alchemy import sym_tbl
from alchemy.pipeline import SchedPipeline, BeginStepPipeline, EndStepPipeline


@SchedPipeline.register()
class LogLRESPipeline(EndStepPipeline):
    def __init__(
        self,
        log_tensorboard: bool = False,
        varname: str = "summary_writer",
        tag: str = "train/lr",
        log_file: bool = False,
        filename: str = "train_lr.log",
        **kwargs
    ) -> None:
        super().__init__()
        self.log_tensorboard = log_tensorboard
        self.varname = varname
        self.tag = tag
        self.log_file = log_file
        self.filename = filename

    def __call__(self, outputs: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        lrs = sym_tbl().optim.get_lr()

        summary_writer: Optional[SummaryWriter] = sym_tbl().try_get_global(self.varname)
        if self.log_tensorboard and summary_writer is not None:
            summary_writer.add_scalar(
                self.tag, lrs[-1], sym_tbl().train_sched.cur_step
            )
        record_dir: Optional[Path] = sym_tbl().record_dir
        if self.log_file and record_dir is not None:
            with (record_dir / self.filename).open('a', encoding="utf8") as f:
                f.write("{}\n".format(",".join(lrs)))
        return kwargs


@SchedPipeline.register()
class LogTrainLossESPipeline(EndStepPipeline):
    def __init__(
        self,
        log_tensorboard: bool = False,
        varname: str = "summary_writer",
        tag: str = "train/loss",
        log_file: bool = False,
        filename: str = "train_loss.log",
        **kwargs
    ) -> None:
        super().__init__()
        self.log_tensorboard = log_tensorboard
        self.varname = varname
        self.tag = tag
        self.log_file = log_file
        self.filename = filename

    def __call__(self, outputs: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        loss = outputs["loss"]

        summary_writer: Optional[SummaryWriter] = sym_tbl().try_get_global(self.varname)
        if self.log_tensorboard and summary_writer is not None:
            summary_writer.add_scalar(
                self.tag, loss, sym_tbl().train_sched.cur_step
            )

        record_dir: Optional[Path] = sym_tbl().record_dir
        if self.log_file and record_dir is not None:
            with (record_dir / self.filename).open('a', encoding="utf8") as f:
                f.write("{}\n".format(loss))
        return kwargs


@SchedPipeline.register()
class ModifyConfigBSPipeline(BeginStepPipeline):
    def __init__(self, step: int, path: List[str], value: Any) -> None:
        super().__init__()
        self.step = step
        self.path = path
        self.value = value

    def __call__(self, batch: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        if sym_tbl().train_sched.cur_step == self.step:
            v = sym_tbl().cfg
            for i, path_item in enumerate(self.path):
                if i == len(self.path) - 1:
                    # last one
                    old_value = v.get(path_item)
                    v[path_item] = self.value
                else:
                    if path_item not in v:
                        v[path_item] = {}
                    v = v[path_item]

            logger.info("Modify config {} from {} to {} at stage {}".format(
                '.'.join(self.path), old_value, self.value,
                sym_tbl().train_sched.cur_step
            ))
        return kwargs


@SchedPipeline.register()
class ResetOptimSchedBSPipeline(BeginStepPipeline):
    def __init__(self, step: int, **kwargs) -> None:
        super().__init__()
        self.step = step
        self.kwargs = kwargs

    def __call__(self, batch: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        if sym_tbl().train_sched.cur_step == self.step:
            logger.info(
                "Reset optimizer and lr_sched with cfg {} at step {}".format(
                    self.kwargs, sym_tbl().train_sched.cur_step
                )
            )

            sym_tbl().optim.reset(**self.kwargs)
            sym_tbl().train_sched.reset_lr_sched(**self.kwargs)

            if sym_tbl().device.type != "cpu":
                with torch.cuda.device(sym_tbl().device):
                    # empty cache uses GPU 0 by default
                    torch.cuda.empty_cache()
        return kwargs
