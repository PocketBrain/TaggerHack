import torch
import torch.nn as nn


def hierarchical_loss(
        y1_true,
        y1_pred,
        y2_true,
        y2_pred,
        y3_true,
        y3_pred,
        parent_mapping,
        class_weights=None,
        epsilon=1e-6,
):
    """
    Вычисляет иерархическую функцию потерь для предсказания на нескольких уровнях иерархии.

    Эта функция рассчитывает потери на трех уровнях иерархии классов. Потери на каждом уровне
    корректируются масками, основанными на правильных предсказаниях родительских классов.
    Потери для уровней 2 и 3 считаются только для тех классов, которые соответствуют предсказанным
    родительским классам.

    Parameters
    ----------
    y1_true : torch.Tensor
        Tensor с истинными значениями меток для первого уровня иерархии (размерности: [batch_size, num_classes_level1]).

    y1_pred : torch.Tensor
        Tensor с предсказанными значениями для первого уровня иерархии (размерности: [batch_size, num_classes_level1]).

    y2_true : torch.Tensor
        Tensor с истинными значениями меток для второго уровня иерархии (размерности: [batch_size, num_classes_level2]).

    y2_pred : torch.Tensor
        Tensor с предсказанными значениями для второго уровня иерархии (размерности: [batch_size, num_classes_level2]).

    y3_true : torch.Tensor
        Tensor с истинными значениями меток для третьего уровня иерархии (размерности: [batch_size, num_classes_level3]).

    y3_pred : torch.Tensor
        Tensor с предсказанными значениями для третьего уровня иерархии (размерности: [batch_size, num_classes_level3]).

    parent_mapping : dict
        Словарь, содержащий сопоставление между классами разных уровней:
        - "l2_to_l1": массив индексов, сопоставляющий классы второго уровня с родительскими классами первого уровня;
        - "l3_to_l2": массив индексов, сопоставляющий классы третьего уровня с родительскими классами второго уровня.

    class_weights : torch.Tensor, optional
        Веса классов для балансировки функции потерь (по умолчанию None).

    epsilon : float, optional
        Малое число для предотвращения деления на ноль (по умолчанию 1e-6).

    Returns
    -------
    total_loss : torch.Tensor
        Общая сумма потерь для всех уровней иерархии, включающая потери на уровнях 1, 2 и 3.

    Notes
    -----
    - Потери на уровнях 2 и 3 рассчитываются только для тех классов, для которых предсказаны
      соответствующие родительские классы на более высоком уровне.
    - Маски valid_positions2 и valid_positions3 используются для обеспечения того, что потери
      на уровнях 2 и 3 учитываются только для валидных позиций.
    - Если нет валидных позиций для какого-либо уровня, потери для этого уровня устанавливаются
      равными нулю.
    """

    criterion = nn.BCELoss(reduction="none")

    # Уровень 1: расчет потерь
    L1_loss = criterion(y1_pred, y1_true)
    L1 = L1_loss.mean()

    # Уровень 2: расчет потерь с учетом родительских классов
    l2_to_l1 = parent_mapping["l2_to_l1"]
    valid_l2 = l2_to_l1 != -1

    mask2 = torch.zeros_like(y2_true)
    if valid_l2.any():
        mask2[:, valid_l2] = y1_true[:, l2_to_l1[valid_l2]]

    valid_positions2 = mask2 > 0
    if valid_positions2.sum() > 0:
        L2_loss = criterion(y2_pred, y2_true)
        L2_loss = L2_loss * valid_positions2.float()
        L2 = L2_loss.sum() / (valid_positions2.float().sum() + epsilon)
    else:
        L2 = torch.tensor(0.0, device=y1_true.device)

    # Уровень 3: расчет потерь с учетом родительских классов
    l3_to_l2 = parent_mapping["l3_to_l2"]
    valid_l3 = l3_to_l2 != -1

    mask3 = torch.zeros_like(y3_true)
    if valid_l3.any():
        mask3[:, valid_l3] = y2_true[:, l3_to_l2[valid_l3]]

    valid_positions3 = mask3 > 0
    if valid_positions3.sum() > 0:
        L3_loss = criterion(y3_pred, y3_true)
        L3_loss = L3_loss * valid_positions3.float()
        L3 = L3_loss.sum() / (valid_positions3.float().sum() + epsilon)
    else:
        L3 = torch.tensor(0.0, device=y1_true.device)

    # Общая сумма потерь
    total_loss = L1 + L2 + L3
    return total_loss
