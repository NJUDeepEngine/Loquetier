from transformers.trainer import *
from transformers.trainer import _is_peft_model

from functools import partial
from typing import Generator
from typing_extensions import LiteralString

from .enums import TrainerProcessType

def train(
    trainer: Trainer,
    resume_from_checkpoint: Optional[Union[str, bool]] = None,
    trial: Optional[Union["optuna.Trial", Dict[str, Any]]] = None,
    ignore_keys_for_eval: Optional[List[str]] = None,
    **kwargs,
) -> Generator[Tuple[TrainerProcessType, Union[Dict[str, Union[torch.Tensor, Any]], torch.Tensor]],
               Optional[torch.Tensor],
               TrainOutput]:
    '''
    Copied from transformers > trainer.py > `Trainer`.

    Preprocess of training loop.
    '''
    if is_sagemaker_mp_enabled():
        raise RuntimeError('Lora Hub Framework doesn\'t support sagemaker mp.')
    if trainer.use_apex:
        raise RuntimeError('Lora Hub Framework doesn\'t support apex.')
    if trainer.args.n_gpu > 1:
        raise RuntimeError('Lora Hub Framework doesn\'t support to train on multiple gpus.')
    
    if resume_from_checkpoint is False:
        resume_from_checkpoint = None

    # memory metrics - must set up as early as possible
    trainer._memory_tracker.start()

    args = trainer.args

    trainer.is_in_train = True

    # Attach NEFTune hooks if necessary
    if trainer.neftune_noise_alpha is not None:
        trainer.model = trainer._activate_neftune(trainer.model)

    # do_train is not a reliable argument, as it might not be set and .train() still called, so
    # the following is a workaround:
    if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
        trainer._move_model_to_device(trainer.model, args.device)

    if "model_path" in kwargs:
        resume_from_checkpoint = kwargs.pop("model_path")
        warnings.warn(
            "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
            "instead.",
            FutureWarning,
        )
    if len(kwargs) > 0:
        raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
    # This might change the seed so needs to run first.
    trainer._hp_search_setup(trial)
    trainer._train_batch_size = trainer.args.train_batch_size

    # Model re-init
    model_reloaded = False
    if trainer.model_init is not None:
        # Seed must be set before instantiating the model when using model_init.
        enable_full_determinism(trainer.args.seed) if trainer.args.full_determinism else set_seed(trainer.args.seed)
        trainer.model = trainer.call_model_init(trial)
        model_reloaded = True
        # Reinitializes optimizer and scheduler
        trainer.optimizer, trainer.lr_scheduler = None, None

    # Load potential model checkpoint
    if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
        resume_from_checkpoint = get_last_checkpoint(args.output_dir)
        if resume_from_checkpoint is None:
            raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

    if resume_from_checkpoint is not None:
        if not is_sagemaker_mp_enabled() and not trainer.is_deepspeed_enabled and not trainer.is_fsdp_enabled:
            trainer._load_from_checkpoint(resume_from_checkpoint)
        # In case of repeating the find_executable_batch_size, set `trainer._train_batch_size` properly
        state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        if state.train_batch_size is not None:
            trainer._train_batch_size = state.train_batch_size

    # If model was re-initialized, put it on the right device and update trainer.model_wrapped
    if model_reloaded:
        if trainer.place_model_on_device:
            trainer._move_model_to_device(trainer.model, args.device)
        trainer.model_wrapped = trainer.model

    inner_training_loop = find_executable_batch_size(
        partial(training_loop, trainer=trainer), trainer._train_batch_size, args.auto_find_batch_size
    )
    if args.push_to_hub:
        try:
            # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
            hf_hub_utils.disable_progress_bars()
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
        finally:
            hf_hub_utils.enable_progress_bars()
    else:
        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

def training_loop(
    trainer: Trainer,
    batch_size: Optional[int] = None,
    args: Optional[TrainingArguments] = None,
    resume_from_checkpoint: Optional[Union[bytes, LiteralString, str]] = None,
    trial: Optional[Union["optuna.Trial", Dict[str, Any]]] = None,
    ignore_keys_for_eval: Optional[List[str]] = None
) -> Generator[Tuple[TrainerProcessType, Union[Dict[str, Union[torch.Tensor, Any]], torch.Tensor]],
               Optional[torch.Tensor],
               TrainOutput]:
    '''
    Copied from transformers > trainer.py > `Trainer`.

    Main training loop.
    '''
    trainer.accelerator.free_memory()
    trainer._train_batch_size = batch_size
    if trainer.args.auto_find_batch_size:
        if trainer.state.train_batch_size != trainer._train_batch_size:
            from accelerate.utils import release_memory

            (trainer.model_wrapped,) = release_memory(trainer.model_wrapped)
            trainer.model_wrapped = trainer.model

            # Check for DeepSpeed *after* the intial pass and modify the config
            if trainer.is_deepspeed_enabled:
                # Temporarily unset `trainer.args.train_batch_size`
                original_bs = trainer.args.per_device_train_batch_size
                trainer.args.per_device_train_batch_size = trainer._train_batch_size // max(1, trainer.args.n_gpu)
                trainer.propagate_args_to_deepspeed(True)
                trainer.args.per_device_train_batch_size = original_bs
        trainer.state.train_batch_size = trainer._train_batch_size
    logger.debug(f"Currently training with a batch size of: {trainer._train_batch_size}")
    # Data loader and number of training steps
    train_dataloader = trainer.get_train_dataloader()
    if trainer.is_fsdp_xla_v2_enabled:
        train_dataloader = tpu_spmd_dataloader(train_dataloader)

    # Setting up training control variables:
    # number of training epochs: num_train_epochs
    # number of training steps per epoch: num_update_steps_per_epoch
    # total number of training steps to execute: max_steps
    total_train_batch_size = trainer._train_batch_size * args.gradient_accumulation_steps * args.world_size

    len_dataloader = None
    num_train_tokens = None
    if has_length(train_dataloader):
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        num_examples = trainer.num_examples(train_dataloader)
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )
            # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
            # the best we can do.
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = (
                    trainer.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                )
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = trainer.num_examples(train_dataloader) * args.num_train_epochs
            if args.include_tokens_per_second:
                num_train_tokens = trainer.num_tokens(train_dataloader) * args.num_train_epochs
    elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
        max_steps = args.max_steps
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_train_epochs = sys.maxsize
        num_update_steps_per_epoch = max_steps
        num_examples = total_train_batch_size * args.max_steps
        num_train_samples = args.max_steps * total_train_batch_size
        if args.include_tokens_per_second:
            num_train_tokens = trainer.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
    else:
        raise ValueError(
            "args.max_steps must be set to a positive value if dataloader does not have a length, was"
            f" {args.max_steps}"
        )

    if DebugOption.UNDERFLOW_OVERFLOW in trainer.args.debug:
        if trainer.args.n_gpu > 1:
            # nn.DataParallel(model) replicates the model, creating new variables and module
            # references registered here no longer work on other gpus, breaking the module
            raise ValueError(
                "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                " (torchrun or torch.distributed.launch (deprecated))."
            )
        else:
            debug_overflow = DebugUnderflowOverflow(trainer.model)  # noqa

    delay_optimizer_creation = is_sagemaker_mp_enabled() or trainer.is_fsdp_xla_enabled or trainer.is_fsdp_enabled

    # We need to reset the scheduler, as its parameters may be different on subsequent calls
    if trainer._created_lr_scheduler:
        trainer.lr_scheduler = None
        trainer._created_lr_scheduler = False

    if trainer.is_deepspeed_enabled:
        trainer.optimizer, trainer.lr_scheduler = deepspeed_init(trainer, num_training_steps=max_steps)

    if not delay_optimizer_creation:
        trainer.create_optimizer_and_scheduler(num_training_steps=max_steps)

    trainer.state = TrainerState(
        stateful_callbacks=[
            cb for cb in trainer.callback_handler.callbacks + [trainer.control] if isinstance(cb, ExportableState)
        ]
    )
    trainer.state.is_hyper_param_search = trial is not None
    trainer.state.train_batch_size = trainer._train_batch_size

    # Compute absolute values for logging, eval, and save if given as ratio
    if args.logging_steps is not None:
        if args.logging_steps < 1:
            trainer.state.logging_steps = math.ceil(max_steps * args.logging_steps)
        else:
            trainer.state.logging_steps = args.logging_steps
    if args.eval_steps is not None:
        if args.eval_steps < 1:
            trainer.state.eval_steps = math.ceil(max_steps * args.eval_steps)
        else:
            trainer.state.eval_steps = args.eval_steps
    if args.save_steps is not None:
        if args.save_steps < 1:
            trainer.state.save_steps = math.ceil(max_steps * args.save_steps)
        else:
            trainer.state.save_steps = args.save_steps

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        if args.gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        else:
            gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

        trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    model = trainer._wrap_model(trainer.model_wrapped)

    # as the model is wrapped, don't use `accelerator.prepare`
    # this is for unhandled cases such as
    # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
    use_accelerator_prepare = True if model is trainer.model else False

    if delay_optimizer_creation:
        if use_accelerator_prepare:
            trainer._fsdp_qlora_plugin_updates()
            trainer.model = trainer.accelerator.prepare(trainer.model)
        trainer.create_optimizer_and_scheduler(num_training_steps=max_steps)

    # prepare using `accelerator` prepare
    if use_accelerator_prepare:
        if hasattr(trainer.lr_scheduler, "step"):
            if trainer.use_apex:
                model = trainer.accelerator.prepare(trainer.model)
            else:
                model, trainer.optimizer = trainer.accelerator.prepare(trainer.model, trainer.optimizer)
        else:
            # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
            model, trainer.optimizer, trainer.lr_scheduler = trainer.accelerator.prepare(
                trainer.model, trainer.optimizer, trainer.lr_scheduler
            )
    elif trainer.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        # In this case we are in DDP + LOMO, which should be supported
        trainer.optimizer = trainer.accelerator.prepare(trainer.optimizer)

    if trainer.is_fsdp_enabled:
        trainer.model = trainer.model_wrapped = model

    # for the rest of this function `model` is the outside model, whether it was wrapped or not
    if model is not trainer.model:
        raise RuntimeError(
            'Expect model is same as trainer.model, but got:\n'
            f'model: {model}\ntrainer.model: {trainer.model}'
        )

    # backward compatibility
    if trainer.is_deepspeed_enabled:
        trainer.deepspeed = trainer.model_wrapped

    # ckpt loading
    if resume_from_checkpoint is not None:
        if trainer.is_deepspeed_enabled:
            deepspeed_load_checkpoint(
                trainer.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(trainer.model)
            )
        elif is_sagemaker_mp_enabled() or trainer.is_fsdp_enabled:
            trainer._load_from_checkpoint(resume_from_checkpoint, trainer.model_wrapped)

    # Check if saved optimizer or scheduler states exist
    trainer._load_optimizer_and_scheduler(resume_from_checkpoint)

    # important: at this point:
    # trainer.model         is the Transformers Model
    # trainer.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
    # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples:,}")
    logger.info(f"  Num Epochs = {num_train_epochs:,}")
    logger.info(f"  Instantaneous batch size per device = {trainer.args.per_device_train_batch_size:,}")
    if trainer.args.per_device_train_batch_size != trainer._train_batch_size:
        logger.info(f"  Training with DataParallel so batch size has been adjusted to: {trainer._train_batch_size:,}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps:,}")
    logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

    trainer.state.epoch = 0
    start_time = time.time()
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    steps_trained_progress_bar = None

    # Check if continuing training from a checkpoint
    if resume_from_checkpoint is not None and os.path.isfile(
        os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    ):
        trainer.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        trainer.compare_trainer_and_checkpoint_args(trainer.args, trainer.state)
        trainer._load_callback_state()
        epochs_trained = int(trainer.state.global_step // num_update_steps_per_epoch)
        if not args.ignore_data_skip:
            steps_trained_in_current_epoch = trainer.state.global_step % (num_update_steps_per_epoch)
            steps_trained_in_current_epoch *= args.gradient_accumulation_steps
        else:
            steps_trained_in_current_epoch = 0

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {trainer.state.global_step}")
        if not args.ignore_data_skip:
            logger.info(
                f"  Will skip the first {epochs_trained} epochs then the first"
                f" {steps_trained_in_current_epoch} batches in the first epoch."
            )

    # Update the references
    trainer.callback_handler.model = trainer.model
    trainer.callback_handler.optimizer = trainer.optimizer
    trainer.callback_handler.lr_scheduler = trainer.lr_scheduler
    trainer.callback_handler.train_dataloader = train_dataloader
    if trainer.hp_name is not None and trainer._trial is not None:
        # use trainer._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
        # parameter to Train when using DDP.
        trainer.state.trial_name = trainer.hp_name(trainer._trial)
    if trial is not None:
        assignments = trial.assignments if trainer.hp_search_backend == HPSearchBackend.SIGOPT else trial
        trainer.state.trial_params = hp_params(assignments)
    else:
        trainer.state.trial_params = None
    # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    # to set this after the load.
    trainer.state.max_steps = max_steps
    trainer.state.num_train_epochs = num_train_epochs
    trainer.state.is_local_process_zero = trainer.is_local_process_zero()
    trainer.state.is_world_process_zero = trainer.is_world_process_zero()

    # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    tr_loss = torch.tensor(0.0).to(args.device)
    # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    trainer._total_loss_scalar = 0.0
    trainer._globalstep_last_logged = trainer.state.global_step
    # TODO
    # model.zero_grad()
    grad_norm: Optional[float] = None
    trainer.control = trainer.callback_handler.on_train_begin(args, trainer.state, trainer.control)

    if args.eval_on_start:
        trainer._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

    total_batched_samples = 0
    for epoch in range(epochs_trained, num_train_epochs):
        epoch_iterator = train_dataloader
        if hasattr(epoch_iterator, "set_epoch"):
            epoch_iterator.set_epoch(epoch)

        # Reset the past mems state at the beginning of each epoch if necessary.
        if args.past_index >= 0:
            trainer._past = None

        steps_in_epoch = (
            len(epoch_iterator)
            if len_dataloader is not None
            else args.max_steps * args.gradient_accumulation_steps
        )
        trainer.control = trainer.callback_handler.on_epoch_begin(args, trainer.state, trainer.control)

        if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
            trainer._load_rng_state(resume_from_checkpoint)

        rng_to_sync = False
        steps_skipped = 0
        if steps_trained_in_current_epoch > 0:
            epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
            steps_skipped = steps_trained_in_current_epoch
            steps_trained_in_current_epoch = 0
            rng_to_sync = True

        step = -1
        for step, inputs in enumerate(epoch_iterator):
            total_batched_samples += 1

            if trainer.args.include_num_input_tokens_seen:
                main_input_name = getattr(trainer.model, "main_input_name", "input_ids")
                if main_input_name not in inputs:
                    logger.warning(
                        "Tried to track the number of tokens seen, however the current model is "
                        "not configured properly to know what item is the input. To fix this, add "
                        "a `main_input_name` attribute to the model class you are using."
                    )
                else:
                    trainer.state.num_input_tokens_seen += (
                        torch.sum(
                            trainer.accelerator.gather(
                                torch.tensor(
                                    inputs[main_input_name].numel(), device=trainer.args.device, dtype=torch.int64
                                )
                            )
                        )
                        .cpu()
                        .item()
                    )
            if rng_to_sync:
                trainer._load_rng_state(resume_from_checkpoint)
                rng_to_sync = False

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                if steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.update(1)
                if steps_trained_in_current_epoch == 0:
                    trainer._load_rng_state(resume_from_checkpoint)
                continue
            elif steps_trained_progress_bar is not None:
                steps_trained_progress_bar.close()
                steps_trained_progress_bar = None

            if step % args.gradient_accumulation_steps == 0:
                trainer.control = trainer.callback_handler.on_step_begin(args, trainer.state, trainer.control)

            with trainer.accelerator.accumulate(model):
                tr_loss_step = yield from training_step(trainer, model, inputs)

            if (
                args.logging_nan_inf_filter
                and not is_torch_xla_available()
                and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
            ):
                # if loss is nan or inf simply add the average of previous logged losses
                tr_loss += tr_loss / (1 + trainer.state.global_step - trainer._globalstep_last_logged)
            else:
                if tr_loss.device != tr_loss_step.device:
                    raise ValueError(
                        f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                    )
                tr_loss += tr_loss_step

            trainer.current_flos += float(trainer.floating_point_ops(inputs))

            is_last_step_and_steps_less_than_grad_acc = (
                steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
            )

            if (
                total_batched_samples % args.gradient_accumulation_steps == 0
                or
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                is_last_step_and_steps_less_than_grad_acc
            ):
                # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                # in accelerate. So, explicitly enable sync gradients to True in that case.
                if is_last_step_and_steps_less_than_grad_acc:
                    trainer.accelerator.gradient_state._set_sync_gradients(True)

                # Gradient clipping
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    # deepspeed does its own clipping

                    if is_sagemaker_mp_enabled() and args.fp16:
                        _grad_norm = trainer.optimizer.clip_master_grads(args.max_grad_norm)
                    elif trainer.use_apex:
                        # Revert to normal clipping otherwise, handling Apex or full precision
                        _grad_norm = nn.utils.clip_grad_norm_(
                            amp.master_params(trainer.optimizer),
                            args.max_grad_norm,
                        )
                    else:
                        _grad_norm = trainer.accelerator.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                        )

                    if (
                        is_accelerate_available()
                        and trainer.accelerator.distributed_type == DistributedType.DEEPSPEED
                    ):
                        grad_norm = model.get_global_grad_norm()
                        # In some cases the grad norm may not return a float
                        if hasattr(grad_norm, "item"):
                            grad_norm = grad_norm.item()
                    else:
                        grad_norm = _grad_norm

                trainer.optimizer.step()

                trainer.control = trainer.callback_handler.on_optimizer_step(args, trainer.state, trainer.control)

                optimizer_was_run = not trainer.accelerator.optimizer_step_was_skipped
                if optimizer_was_run:
                    # Delay optimizer scheduling until metrics are generated
                    if not isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        trainer.lr_scheduler.step()

                # TODO
                # model.zero_grad()
                trainer.state.global_step += 1
                trainer.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                trainer.control = trainer.callback_handler.on_step_end(args, trainer.state, trainer.control)

                yield from maybe_log_save_evaluate(trainer, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
            else:
                trainer.control = trainer.callback_handler.on_substep_end(args, trainer.state, trainer.control)

            if trainer.control.should_epoch_stop or trainer.control.should_training_stop:
                # PyTorch/XLA relies on the data loader to insert the mark_step for
                # each step. Since we are breaking the loop early, we need to manually
                # insert the mark_step here.
                if is_torch_xla_available():
                    xm.mark_step()
                break
        if step < 0:
            logger.warning(
                "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                f" {trainer.state.global_step}! This is expected if you're using an IterableDataset and set"
                f" num_steps ({max_steps}) higher than the number of available samples."
            )
            trainer.control.should_training_stop = True

        trainer.control = trainer.callback_handler.on_epoch_end(args, trainer.state, trainer.control)
        
        yield from maybe_log_save_evaluate(trainer, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

        if DebugOption.TPU_METRICS_DEBUG in trainer.args.debug:
            if is_torch_xla_available():
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())
            else:
                logger.warning(
                    "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                    "configured. Check your training configuration if this is unexpected."
                )
        if trainer.control.should_training_stop:
            break

    if args.past_index and hasattr(trainer, "_past"):
        # Clean the state at the end of training
        delattr(trainer, "_past")

    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    if args.load_best_model_at_end and trainer.state.best_model_checkpoint is not None:
        # Wait for everyone to get here so we are sure the model has been saved by process 0.
        if is_torch_xla_available():
            xm.rendezvous("load_best_model_at_end")
        elif args.parallel_mode == ParallelMode.DISTRIBUTED:
            dist.barrier()
        elif is_sagemaker_mp_enabled():
            smp.barrier()

        trainer._load_best_model()

    # add remaining tr_loss
    trainer._total_loss_scalar += tr_loss.item()
    effective_global_step = max(trainer.state.global_step, 0.001)  # Avoid ZeroDivisionError
    train_loss = trainer._total_loss_scalar / effective_global_step

    metrics = speed_metrics(
        "train",
        start_time,
        num_samples=num_train_samples,
        num_steps=trainer.state.max_steps,
        num_tokens=num_train_tokens,
    )
    trainer.store_flos()
    metrics["total_flos"] = trainer.state.total_flos
    metrics["train_loss"] = train_loss

    trainer.is_in_train = False

    trainer._memory_tracker.stop_and_update_metrics(metrics)

    trainer.log(metrics)

    run_dir = trainer._get_output_dir(trial)
    checkpoints_sorted = trainer._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

    # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    if trainer.args.should_save and trainer.state.best_model_checkpoint is not None and trainer.args.save_total_limit == 1:
        for checkpoint in checkpoints_sorted:
            if not os.path.samefile(checkpoint, trainer.state.best_model_checkpoint):
                logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                shutil.rmtree(checkpoint, ignore_errors=True)

    trainer.control = trainer.callback_handler.on_train_end(args, trainer.state, trainer.control)

    # Wait for the checkpoint to be uploaded.
    trainer._finish_current_push()

    # After training we make sure to retrieve back the original forward pass method
    # for the embedding layer by removing the forward post hook.
    if trainer.neftune_noise_alpha is not None:
        trainer._deactivate_neftune(trainer.model)

    return TrainOutput(trainer.state.global_step, train_loss, metrics)

def training_loop_posthook(func_list: List[Callable], exec_func: Optional[Callable] = None):
    '''
    Post-process of training loop.
    '''
    res = None
    if exec_func is not None:
        res = exec_func()
    for func in func_list:
        func()
    return res

def training_step(
    trainer: Trainer,
    model: nn.Module,
    inputs: Dict[str, Union[torch.Tensor, Any]]
) -> Generator[Tuple[TrainerProcessType, Union[Dict[str, Union[torch.Tensor, Any]], torch.Tensor]],
               Optional[torch.Tensor],
               torch.Tensor]:
    '''
    Copied from transformers > trainer.py > `Trainer`.

    Perform training step.
    '''
    inputs = trainer._prepare_inputs(inputs)
    if is_sagemaker_mp_enabled():
        loss_mb = smp_forward_backward(model, inputs, trainer.args.gradient_accumulation_steps)
        return loss_mb.reduce_mean().detach().to(trainer.args.device)

    with trainer.compute_loss_context_manager():
        loss = yield from compute_loss(trainer, model, inputs)

    del inputs
    if (
        trainer.args.torch_empty_cache_steps is not None
        and trainer.state.global_step % trainer.args.torch_empty_cache_steps == 0
    ):
        if is_xpu_available():
            torch.xpu.empty_cache()
        elif is_mlu_available():
            torch.mlu.empty_cache()
        elif is_npu_available():
            torch.npu.empty_cache()
        elif is_torch_version(">=", "2.0") and is_mps_available():
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()

    kwargs = {}

    # For LOMO optimizers you need to explicitly use the learnign rate
    if trainer.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        kwargs["learning_rate"] = trainer._get_learning_rate()

    if trainer.args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training

    if trainer.use_apex:
        with amp.scale_loss(loss, trainer.optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        yield TrainerProcessType.Backward, loss

    return loss.detach()

def compute_loss(
    trainer: Trainer,
    model: nn.Module,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    return_outputs: bool = False,
    forward_type: TrainerProcessType = TrainerProcessType.TrainForward
) -> Generator[Tuple[TrainerProcessType, Dict[str, Union[torch.Tensor, Any]]],
               Tuple[torch.Tensor, Optional[torch.Tensor]],
               Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
    '''
    Copied from transformers > trainer.py > `Trainer`.

    Customized loss computation.
    '''
    if trainer.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    outputs = yield forward_type, inputs
    
    if trainer.args.past_index >= 0:
        trainer._past = outputs[trainer.args.past_index]

    if labels is not None:
        unwrapped_model = trainer.accelerator.unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = trainer.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = trainer.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    return (loss, outputs) if return_outputs else loss

def maybe_log_save_evaluate(
    trainer: Trainer,
    tr_loss: torch.Tensor,
    grad_norm: Optional[Union[torch.Tensor, float]],
    model: nn.Module,
    trial: Optional[Union["optuna.Trial", Dict[str, Any]]],
    epoch: int,
    ignore_keys_for_eval
) -> Generator[Tuple[TrainerProcessType, Dict[str, Union[torch.Tensor, Any]]],
               Dict[str, float],
               None]:
    '''
    Copied from transformers > trainer.py > `Trainer`.

    Perform log save evaluate.
    '''
    if trainer.control.should_log and trainer.state.global_step > trainer._globalstep_last_logged:
        if is_torch_xla_available():
            xm.mark_step()

        logs: Dict[str, float] = {}

        # all_gather + mean() to get average loss over all processes
        tr_loss_scalar = trainer._nested_gather(tr_loss).mean().item()

        # reset tr_loss to zero
        tr_loss -= tr_loss

        logs["loss"] = round(tr_loss_scalar / (trainer.state.global_step - trainer._globalstep_last_logged), 4)
        if grad_norm is not None:
            logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        logs["learning_rate"] = trainer._get_learning_rate()

        trainer._total_loss_scalar += tr_loss_scalar
        trainer._globalstep_last_logged = trainer.state.global_step
        trainer.store_flos()

        trainer.log(logs)

    metrics = None
    if trainer.control.should_evaluate:
        metrics = yield from evaluate_with_check(trainer, trial, ignore_keys_for_eval)

    if trainer.control.should_save:
        trainer._save_checkpoint(model, trial, metrics=metrics)
        trainer.control = trainer.callback_handler.on_save(trainer.args, trainer.state, trainer.control)

def evaluate_with_check(
    trainer: Trainer,
    trial: Optional[Union["optuna.Trial", Dict[str, Any]]],
    ignore_keys_for_eval: Optional[List[str]],
    skip_scheduler: bool = False
) -> Generator[Tuple[TrainerProcessType, Dict[str, Union[torch.Tensor, Any]]],
               Dict[str, float],
               Dict[str, float]]:
    '''
    Copied from transformers > trainer.py > `Trainer`.

    Perform evaluate with report and schedule.
    '''
    metrics = yield from evaluate(trainer, ignore_keys=ignore_keys_for_eval)
    trainer._report_to_hp_search(trial, trainer.state.global_step, metrics)

    # Run delayed LR scheduler now that metrics are populated
    if isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and not skip_scheduler:
        metric_to_check = trainer.args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        try:
            trainer.lr_scheduler.step(metrics[metric_to_check])
        except KeyError as exc:
            raise KeyError(
                f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
            ) from exc
    return metrics

def evaluate(
    trainer: Trainer,
    eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
    ignore_keys: Optional[List[str]] = None,
    metric_key_prefix: str = 'eval'
) -> Generator[Tuple[TrainerProcessType, Dict[str, Union[torch.Tensor, Any]]],
               Union[EvalLoopOutput, Dict[str, float]],
               Dict[str, float]]:
    '''
    Copied from transformers > trainer.py > `Trainer`.

    Evaluate.
    '''
    # handle multipe eval datasets
    override = eval_dataset is not None
    eval_dataset = eval_dataset if override else trainer.eval_dataset
    if isinstance(eval_dataset, dict):
        metrics = {}
        for eval_dataset_name, _eval_dataset in eval_dataset.items():
            dataset_metrics = yield from evaluate(
                trainer,
                eval_dataset=_eval_dataset if override else eval_dataset_name,
                ignore_keys=ignore_keys,
                metric_key_prefix=f'{metric_key_prefix}_{eval_dataset_name}',
            )
            metrics.update(dataset_metrics)
        return metrics

    # memory metrics - must set up as early as possible
    trainer._memory_tracker.start()

    eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
    if trainer.is_fsdp_xla_v2_enabled:
        eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

    start_time = time.time()

    
    output = yield from evaluation_loop(
        trainer,
        eval_dataloader,
        description='Evaluation',
        prediction_loss_only=True if trainer.compute_metrics is None else None,
        ignore_keys=ignore_keys,
        metric_key_prefix=metric_key_prefix,
    )

    total_batch_size = trainer.args.eval_batch_size * trainer.args.world_size
    if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
        start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
    if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
        start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
    output.metrics.update(
        speed_metrics(
            metric_key_prefix,
            start_time,
            num_samples=output.num_samples,
            num_steps=math.ceil(output.num_samples / total_batch_size),
        )
    )

    trainer.log(output.metrics)

    if DebugOption.TPU_METRICS_DEBUG in trainer.args.debug:
        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        xm.master_print(met.metrics_report())

    trainer.control = trainer.callback_handler.on_evaluate(trainer.args, trainer.state, trainer.control, output.metrics)

    trainer._memory_tracker.stop_and_update_metrics(output.metrics)

    return output.metrics

def evaluation_loop(
    trainer: Trainer,
    dataloader: DataLoader,
    description: str,
    prediction_loss_only: Optional[bool] = None,
    ignore_keys: Optional[List[str]] = None,
    metric_key_prefix: str = 'eval'
) -> Generator[Tuple[TrainerProcessType, Dict[str, Union[torch.Tensor, Any]]],
               Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
               EvalLoopOutput]:
    '''
    Copied from transformers > trainer.py > `Trainer`.

    Evaluation loop.
    '''
    args = trainer.args

    prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

    # if eval is called w/o train, handle model prep here
    if trainer.is_deepspeed_enabled and trainer.deepspeed is None:
        _, _ = deepspeed_init(trainer, num_training_steps=0, inference=True)

    model = trainer._wrap_model(trainer.model, training=False, dataloader=dataloader)

    if len(trainer.accelerator._models) == 0 and model is trainer.model:
        start_time = time.time()
        model = (
            trainer.accelerator.prepare(model)
            if trainer.is_deepspeed_enabled
            else trainer.accelerator.prepare_model(model, evaluation_mode=True)
        )
        trainer.model_preparation_time = round(time.time() - start_time, 4)

        if trainer.is_fsdp_enabled:
            trainer.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not trainer.model:
            trainer.model_wrapped = model

        # backward compatibility
        if trainer.is_deepspeed_enabled:
            trainer.deepspeed = trainer.model_wrapped

    # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
    # while ``train`` is running, cast it to the right dtype first and then put on device
    if not trainer.is_in_train:
        if args.fp16_full_eval:
            model = model.to(dtype=torch.float16, device=args.device)
        elif args.bf16_full_eval:
            model = model.to(dtype=torch.bfloat16, device=args.device)

    batch_size = trainer.args.eval_batch_size

    logger.info(f"\n***** Running {description} *****")
    if has_length(dataloader):
        logger.info(f"  Num examples = {trainer.num_examples(dataloader)}")
    else:
        logger.info("  Num examples: Unknown")
    logger.info(f"  Batch size = {batch_size}")

    trainer.callback_handler.eval_dataloader = dataloader
    # Do this before wrapping.
    eval_dataset = getattr(dataloader, "dataset", None)

    if args.past_index >= 0:
        trainer._past = None

    # Initialize containers
    all_losses = EvalLoopContainer(trainer.args.eval_do_concat_batches, padding_index=-100)
    all_preds = EvalLoopContainer(trainer.args.eval_do_concat_batches, padding_index=-100)
    all_labels = EvalLoopContainer(trainer.args.eval_do_concat_batches, padding_index=-100)
    all_inputs = EvalLoopContainer(trainer.args.eval_do_concat_batches, padding_index=-100)

    metrics = None

    # Will be useful when we have an iterable dataset so don't know its length.
    observed_num_examples = 0

    # Main evaluation loop
    for step, inputs in enumerate(dataloader):
        # Update the observed num examples
        observed_batch_size = find_batch_size(inputs)
        if observed_batch_size is not None:
            observed_num_examples += observed_batch_size
            # For batch samplers, batch_size is not known by the dataloader in advance.
            if batch_size is None:
                batch_size = observed_batch_size

        # Prediction step
        losses, logits, labels = yield from prediction_step(
            trainer, model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )
        main_input_name = getattr(trainer.model, "main_input_name", "input_ids")
        inputs_decode = trainer._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

        if is_torch_xla_available():
            xm.mark_step()

        # Update containers
        if losses is not None:
            losses = trainer.gather_function((losses.repeat(batch_size)))
            all_losses.add(losses)
        if inputs_decode is not None:
            inputs_decode = trainer.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
            inputs_decode = trainer.gather_function((inputs_decode))
            if not trainer.args.batch_eval_metrics or description == "Prediction":
                all_inputs.add(inputs_decode)
        if labels is not None:
            # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
            labels = trainer.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        if logits is not None:
            logits = trainer.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
            if trainer.preprocess_logits_for_metrics is not None:
                logits = trainer.preprocess_logits_for_metrics(logits, labels)
            logits = trainer.gather_function((logits))
            if not trainer.args.batch_eval_metrics or description == "Prediction":
                all_preds.add(logits)
        if labels is not None:
            labels = trainer.gather_function((labels))
            if not trainer.args.batch_eval_metrics or description == "Prediction":
                all_labels.add(labels)

        trainer.control = trainer.callback_handler.on_prediction_step(args, trainer.state, trainer.control)

        if trainer.args.batch_eval_metrics:
            if trainer.compute_metrics is not None and logits is not None and labels is not None:
                is_last_step = trainer.accelerator.gradient_state.end_of_dataloader
                if args.include_inputs_for_metrics:
                    metrics = trainer.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, inputs=inputs),
                        compute_result=is_last_step,
                    )
                else:
                    metrics = trainer.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels),
                        compute_result=is_last_step,
                    )

            del losses, logits, labels, inputs
            torch.cuda.empty_cache()

        # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
        elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
            all_losses.to_cpu_and_numpy()
            all_preds.to_cpu_and_numpy()
            all_labels.to_cpu_and_numpy()
            all_inputs.to_cpu_and_numpy()

            del losses, logits, labels, inputs
            torch.cuda.empty_cache()

    # After all calls to `.gather_function`, reset to `gather_for_metrics`:
    trainer.gather_function = trainer.accelerator.gather_for_metrics
    if args.past_index and hasattr(trainer, "_past"):
        # Clean the state at the end of the evaluation loop
        delattr(trainer, "_past")

    # Gather all remaining tensors and put them back on the CPU
    all_losses = all_losses.get_arrays()
    all_preds = all_preds.get_arrays()
    all_labels = all_labels.get_arrays()
    all_inputs = all_inputs.get_arrays()

    # Number of samples
    if has_length(eval_dataset):
        num_samples = len(eval_dataset)
    # The instance check is weird and does not actually check for the type, but whether the dataset has the right
    # methods. Therefore we need to make sure it also has the attribute.
    elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
        num_samples = eval_dataset.num_examples
    else:
        if has_length(dataloader):
            num_samples = trainer.num_examples(dataloader)
        else:  # both len(dataloader.dataset) and len(dataloader) fail
            num_samples = observed_num_examples
    if num_samples == 0 and observed_num_examples > 0:
        num_samples = observed_num_examples

    # Metrics!
    if (
        trainer.compute_metrics is not None
        and all_preds is not None
        and all_labels is not None
        and not trainer.args.batch_eval_metrics
    ):
        if args.include_inputs_for_metrics:
            metrics = trainer.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
            )
        else:
            metrics = trainer.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    elif metrics is None:
        metrics = {}

    # To be JSON-serializable, we need to remove numpy types or zero-d tensors
    metrics = denumpify_detensorize(metrics)

    if isinstance(all_losses, list) and all_losses:
        metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
    elif isinstance(all_losses, np.ndarray):
        metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
    if hasattr(trainer, "jit_compilation_time"):
        metrics[f"{metric_key_prefix}_jit_compilation_time"] = trainer.jit_compilation_time
    if hasattr(trainer, "model_preparation_time"):
        metrics[f"{metric_key_prefix}_model_preparation_time"] = trainer.model_preparation_time

    # Prefix all keys with metric_key_prefix + '_'
    for key in list(metrics.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

    return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

def prediction_step(
    trainer: Trainer,
    model: nn.Module,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    prediction_loss_only: bool,
    ignore_keys: Optional[List[str]] = None
) -> Generator[Tuple[TrainerProcessType, Dict[str, Union[torch.Tensor, Any]]],
               Tuple[torch.Tensor, Optional[torch.Tensor]],
               Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]:
    '''
    Copied from transformers > trainer.py > `Trainer`.

    Perform prediction step.
    '''
    has_labels = False if len(trainer.label_names) == 0 else all(inputs.get(k) is not None for k in trainer.label_names)

    return_loss = inputs.get("return_loss", None)
    if return_loss is None:
        return_loss = trainer.can_return_loss
    loss_without_labels = True if len(trainer.label_names) == 0 and return_loss else False

    inputs = trainer._prepare_inputs(inputs)
    if ignore_keys is None:
        if hasattr(trainer.model, "config"):
            ignore_keys = getattr(trainer.model.config, "keys_to_ignore_at_inference", [])
        else:
            ignore_keys = []

    if has_labels or loss_without_labels:
        labels = nested_detach(tuple(inputs.get(name) for name in trainer.label_names))
        if len(labels) == 1:
            labels = labels[0]
    else:
        labels = None

    if has_labels or loss_without_labels:
        with trainer.compute_loss_context_manager():
            loss, outputs = yield from compute_loss(
                trainer, model, inputs,
                return_outputs=True, forward_type=TrainerProcessType.EvaluateForward
            )
        loss = loss.mean().detach()

        if isinstance(outputs, dict):
            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
        else:
            logits = outputs[1:]
    else:
        loss = None
        with trainer.compute_loss_context_manager():
            outputs = yield TrainerProcessType.EvaluateForward, inputs
        if isinstance(outputs, dict):
            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
        else:
            logits = outputs
        if trainer.args.past_index >= 0:
            trainer._past = outputs[trainer.args.past_index - 1]

    if prediction_loss_only:
        return (loss, None, None)

    logits = nested_detach(logits)
    if len(logits) == 1:
        logits = logits[0]

    return (loss, logits, labels)


class WrappedTrainingLoopGenerator:
    def __init__(self, trainer: Trainer, train_kwargs: Dict[str, Any] = {}):
        self.trainer = trainer
        self._is_training_stopped = True
        self._next_value = None
        self._return_value = None
        
        self.training_data_gen = train(self.trainer, **train_kwargs)
        self._init_training_data()
    
    def _init_training_data(self):
        self._is_training_stopped = False
        self.get_next_value()
    
    def get_next_value(self, send_value = None):
        try:
            if send_value is None:
                self._next_value = next(self.training_data_gen)
            else:
                self._next_value = self.training_data_gen.send(send_value)
        except StopIteration as e:
            self._is_training_stopped = True
            self._next_value = None
            self._return_value = e.value
    
    @property
    def is_training_stopped(self) -> bool:
        return self._is_training_stopped
    
    @property
    def next_value(self) -> Optional[Tuple[
        TrainerProcessType,
        Union[Dict[str, Union[torch.Tensor, Any]], torch.Tensor]
    ]]:
        return self._next_value
    
    @property
    def return_value(self) -> Optional[TrainOutput]:
        return self._return_value

