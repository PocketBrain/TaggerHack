import torch.nn as nn

class HierarchicalMultimodalClassifier(nn.Module):
    """
    Иерархический мультимодальный классификатор с тремя уровнями предсказаний.

    Модель использует эмбеддинги в качестве входных данных и выполняет классификацию на трех уровнях
    иерархии. На каждом уровне предсказываются отдельные классы с использованием сигмоидной активации.

    Parameters
    ----------
    embedding_dim : int
        Размерность входных эмбеддингов.
    hidden_dim : int
        Размер скрытого слоя модели.
    level1_size : int
        Количество классов для первого уровня классификации.
    level2_size : int
        Количество классов для второго уровня классификации.
    level3_size : int
        Количество классов для третьего уровня классификации.

    Attributes
    ----------
    fc1 : nn.Linear
        Первый полносвязный слой, преобразующий входные эмбеддинги в скрытое пространство размером (hidden_dim * 2).
    fc2 : nn.Linear
        Второй полносвязный слой, преобразующий скрытые представления в пространство размером (hidden_dim).
    relu : nn.ReLU
        Функция активации ReLU, используемая после каждого полносвязного слоя.
    level1_out : nn.Linear
        Выходной полносвязный слой для первого уровня классификации, который предсказывает классы первого уровня.
    level2_out : nn.Linear
        Выходной полносвязный слой для второго уровня классификации.
    level3_out : nn.Linear
        Выходной полносвязный слой для третьего уровня классификации.
    sigmoid : nn.Sigmoid
        Сигмоидная функция активации, применяемая к выходам каждого уровня для получения вероятностей классов.

    Methods
    -------
    forward(x)
        Прямое распространение (forward pass) через модель.
        Возвращает предсказания для трех уровней иерархии.
    """

    def __init__(
        self, embedding_dim, hidden_dim, level1_size, level2_size, level3_size
    ):
        super(HierarchicalMultimodalClassifier, self).__init__()
        # Первый полносвязный слой
        self.fc1 = nn.Linear(embedding_dim, hidden_dim * 2)
        # Второй полносвязный слой
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        # Функция активации ReLU
        self.relu = nn.ReLU()
        # Выходные слои для предсказаний на разных уровнях иерархии
        self.level1_out = nn.Linear(hidden_dim, level1_size)
        self.level2_out = nn.Linear(hidden_dim, level2_size)
        self.level3_out = nn.Linear(hidden_dim, level3_size)
        # Функция активации Sigmoid для получения вероятностей классов
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Прямое распространение через модель.

        Parameters
        ----------
        x : torch.Tensor
            Входной тензор с эмбеддингами (размерности [batch_size, embedding_dim]).

        Returns
        -------
        y1 : torch.Tensor
            Предсказания классов для первого уровня иерархии (размерности [batch_size, level1_size]).
        y2 : torch.Tensor
            Предсказания классов для второго уровня иерархии (размерности [batch_size, level2_size]).
        y3 : torch.Tensor
            Предсказания классов для третьего уровня иерархии (размерности [batch_size, level3_size]).
        """
        # Проход через первый полносвязный слой и активацию
        x = self.fc1(x)
        x = self.relu(x)
        # Проход через второй полносвязный слой и активацию
        x = self.fc2(x)
        x = self.relu(x)
        # Предсказания для трех уровней с применением сигмоидной активации
        y1 = self.sigmoid(self.level1_out(x))
        y2 = self.sigmoid(self.level2_out(x))
        y3 = self.sigmoid(self.level3_out(x))
        return y1, y2, y3
