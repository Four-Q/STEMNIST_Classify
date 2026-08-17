import torch
from tqdm.auto import tqdm
from spikingjelly.activation_based import functional
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch=None):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 清理 Notebook 之前可能遗留的 LIF 状态
    functional.reset_net(model)

    description = (
        f"Train Epoch {epoch}"
        if epoch is not None
        else "Training"
    )

    progress_bar = tqdm(
        train_loader,
        desc=description,
        dynamic_ncols=True,
        leave=True,
    )

    for inputs, labels in progress_bar:
        # DataLoader:
        # inputs [N,T,C,H,W]
        # labels  [N]
        inputs = inputs.to(
            DEVICE,
            dtype=torch.float32,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE  ,
            dtype=torch.long,
            non_blocking=True,
        )

        # SpikingJelly 多步模式要求 [T,N,C,H,W]
        inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
        batch_size = labels.size(0)

        # 清除上一个批次的膜电位
        functional.reset_net(model)
        # 清除参数梯度
        optimizer.zero_grad(set_to_none=True)

        # logits: [N,num_classes]
        logits = model(inputs)
        loss = criterion(
            logits.float(),
            labels,
        )
        # 反向传播
        loss.backward()
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )
        # 更新参数
        optimizer.step()

        # 保存当前批次的统计值
        batch_loss = loss.detach().item()
        predictions = (
            logits.detach()
            .argmax(dim=1)
        )
        batch_correct = (
            predictions == labels
        ).sum().item()

        total_loss += batch_loss * batch_size
        total_correct += batch_correct
        total_samples += batch_size

        average_loss = total_loss / total_samples
        average_accuracy = total_correct / total_samples

        # 更新进度条右侧信息
        progress_bar.set_postfix(
            loss=f"{average_loss:.4f}",
            accuracy=f"{average_accuracy:.4f}",
        )

    if total_samples == 0:
        raise RuntimeError("train_loader 中没有样本。")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "samples": total_samples,
    }


def validate_epoch(
    model,
    val_loader,
    criterion,
    device,
    epoch=None,
):

    # 保存模型原来的训练/验证状态
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_targets = []
    all_predictions = []

    firing_rate_sums = {
        "lif1": 0.0,
        "lif2": 0.0,
        "output_lif": 0.0,
    }

    description = (
        f"Val Epoch {epoch}"
        if epoch is not None
        else "Validation"
    )

    functional.reset_net(model)

    progress_bar = tqdm(
        val_loader,
        desc=description,
        dynamic_ncols=True,
        leave=True,
    )
    # 验证阶段不需要构建计算图
    with torch.no_grad():
        for inputs, labels in progress_bar:

            inputs = inputs.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                dtype=torch.long,
                non_blocking=True,
            )
            # [N,T,C,H,W] -> [T,N,C,H,W]
            inputs = inputs.permute(
                1, 0, 2, 3, 4
            ).contiguous()

            batch_size = labels.size(0)

            functional.reset_net(model)
            # logits: [N,35]
            logits, firing_rates = model(
                inputs,
                return_firing_rates=True,
            )
            loss = criterion(
                logits.float(),
                labels,
            )

            predictions = logits.argmax(dim=1)
            batch_correct = (
                predictions == labels
            ).sum().item()

            total_loss += loss.item() * batch_size
            total_correct += batch_correct
            total_samples += batch_size

            all_targets.extend(
                labels.cpu().tolist()
            )
            all_predictions.extend(
                predictions.cpu().tolist()
            )

            for name, rate in firing_rates.items():
                firing_rate_sums[name] += (
                    rate.item() * batch_size
                )

            average_loss = total_loss / total_samples
            average_accuracy = (
                total_correct / total_samples
            )

            progress_bar.set_postfix(
                loss=f"{average_loss:.4f}",
                accuracy=f"{average_accuracy:.4f}",
            )

    functional.reset_net(model)

    # 恢复调用验证函数之前的状态
    model.train(was_training)

    if total_samples == 0:
        raise RuntimeError("val_loader 中没有样本。")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "samples": total_samples,
        "firing_rates": {
            name: value / total_samples
            for name, value in firing_rate_sums.items()
        },
    }

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    save_path="best_model.pt",
    scheduler=None
):
    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": []
    }

    best_val_accuracy = -1.0
    best_epoch = 0

    # 防止 Notebook 之前的运行留下膜电位
    functional.reset_net(model)

    for epoch in range(1, num_epochs + 1):
        # 当前学习率
        current_lr = optimizer.param_groups[0]["lr"]

        # 训练
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            DEVICE=device,
            epoch=epoch,
        )

        functional.reset_net(model)

        # 验证
        val_metrics = validate_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        # 发放率监控
        rates = val_metrics["firing_rates"]

        print(
            "Validation firing rates | "
            f"lif1={rates['lif1']:.6f} | "
            f"lif2={rates['lif2']:.6f} | "
            f"output={rates['output_lif']:.6f}"
        )

        # if rates["output_lif"] <= 1e-6:
        #     raise RuntimeError(
        #         "输出 LIF 发放率为 0，网络已经进入全沉默状态。"
        #         "请检查输入尺度、学习率和分类层参数。"
        #     )
        
        functional.reset_net(model)

        # 记录历史数据
        history["train_loss"].append(
            train_metrics["loss"]
        )
        history["train_accuracy"].append(
            train_metrics["accuracy"]
        )
        history["val_loss"].append(
            val_metrics["loss"]
        )
        history["val_accuracy"].append(
            val_metrics["accuracy"]
        )
        history["learning_rate"].append(current_lr)

        train_loss = train_metrics["loss"]
        train_accuracy = train_metrics["accuracy"]
        val_loss = val_metrics["loss"]
        val_accuracy = val_metrics["accuracy"]

        print(
            f"\nEpoch {epoch:03d}/{num_epochs:03d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6g}"
        )

        # 学习率衰退
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # print(
        #     "Validation firing rates:",
        #     val_metrics["firing_rates"],
        # )

        # 根据验证集准确率保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "history": history,
            }

            torch.save(
                checkpoint,
                save_path,
            )

            print(
                f"✓ 保存最佳模型：{save_path}\n"
                f"  epoch={epoch}, "
                f"val_accuracy={val_accuracy:.4f}"
            )

    print(f"\n{'=' * 50}")
    print("训练完成")
    print(f"最佳 epoch：{best_epoch}")
    print(f"最佳验证集准确率：{best_val_accuracy:.4f}")
    print(f"最佳模型路径：{save_path}")
    print(f"{'=' * 50}")

    # 自动载入最佳模型，而不是保留最后一个 epoch 的模型
    best_checkpoint = torch.load(
        save_path,
        map_location=device,
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    functional.reset_net(model)

    return history


#######################################
# 可视化训练history

def plot_training_history(
    history,
    save_path=None,
    show=True,
    dpi=300,
):
    """
    绘制模型训练历史。

    左图：训练集和验证集 loss
    右图：训练集和验证集 accuracy

    参数：
        history:
            train_model() 返回的历史字典，必须包含：
            train_loss、train_accuracy、val_loss、val_accuracy。

        save_path:
            图片保存路径，例如：
            "../outputs/1/training_history.png"
            设为 None 时不保存。

        show:
            是否在 Notebook 中显示图片。

        dpi:
            保存图片的分辨率。

    返回：
        matplotlib Figure 对象。
    """

    required_keys = {
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    }

    missing_keys = required_keys.difference(history)

    if missing_keys:
        raise KeyError(
            f"history 缺少字段：{sorted(missing_keys)}"
        )

    lengths = {
        key: len(history[key])
        for key in required_keys
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"history 中各指标长度不一致：{lengths}"
        )

    num_epochs = lengths["train_loss"]

    if num_epochs == 0:
        raise ValueError("history 中没有训练记录。")

    epochs = range(1, num_epochs + 1)

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 5),
        sharex=True,
    )

    # ========================================================
    # 左图：Loss
    # ========================================================
    loss_axis = axes[0]

    loss_axis.plot(
        epochs,
        history["train_loss"],
        color="tab:red",
        linewidth=1.8,
        label="Train Loss",
    )

    loss_axis.plot(
        epochs,
        history["val_loss"],
        color="tab:orange",
        linewidth=1.8,
        label="Validation Loss",
    )

    minimum_val_loss_index = min(
        range(num_epochs),
        key=lambda index: history["val_loss"][index],
    )
    minimum_val_loss_epoch = minimum_val_loss_index + 1
    minimum_val_loss = history["val_loss"][minimum_val_loss_index]

    loss_axis.scatter(
        minimum_val_loss_epoch,
        minimum_val_loss,
        color="tab:orange",
        s=45,
        zorder=5,
    )
    loss_axis.annotate(
        f"Best val loss: {minimum_val_loss:.3f}\n"
        f"Epoch {minimum_val_loss_epoch}",
        xy=(minimum_val_loss_epoch, minimum_val_loss),
        xytext=(-85, 28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "tab:orange"},
        fontsize=9,
    )

    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Cross-entropy Loss")
    loss_axis.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )
    loss_axis.legend(loc="best")

    # ========================================================
    # 右图：Accuracy
    # ========================================================
    accuracy_axis = axes[1]

    accuracy_axis.plot(
        epochs,
        history["train_accuracy"],
        color="tab:blue",
        linewidth=1.8,
        label="Train Accuracy",
    )

    accuracy_axis.plot(
        epochs,
        history["val_accuracy"],
        color="tab:green",
        linewidth=1.8,
        label="Validation Accuracy",
    )

    best_val_accuracy_index = max(
        range(num_epochs),
        key=lambda index: history["val_accuracy"][index],
    )
    best_val_accuracy_epoch = best_val_accuracy_index + 1
    best_val_accuracy = history["val_accuracy"][best_val_accuracy_index]

    accuracy_axis.scatter(
        best_val_accuracy_epoch,
        best_val_accuracy,
        color="tab:green",
        s=45,
        zorder=5,
    )
    accuracy_axis.annotate(
        f"Best val accuracy: {best_val_accuracy:.1%}\n"
        f"Epoch {best_val_accuracy_epoch}",
        xy=(best_val_accuracy_epoch, best_val_accuracy),
        xytext=(-115, -18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "tab:green"},
        fontsize=9,
    )

    maximum_accuracy = max(
        max(history["train_accuracy"]),
        max(history["val_accuracy"]),
    )
    accuracy_upper_limit = min(
        1.0,
        max(0.1, maximum_accuracy * 1.1),
    )

    accuracy_axis.set_title("Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0.0, accuracy_upper_limit)
    accuracy_axis.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )
    accuracy_axis.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )
    accuracy_axis.legend(loc="best")

    figure.suptitle(
        "SNN Training History",
        fontsize=14,
    )

    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )

        print(f"训练曲线已保存：{save_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(figure)

    return figure
