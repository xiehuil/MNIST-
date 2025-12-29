import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# 1. 加载和预处理数据
def load_and_preprocess_data():
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
    
    # 数据归一化到[0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 添加通道维度
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    
    # 将标签转换为独热编码
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)


# 2. 构建CNN模型
def create_model():
    model = tf.keras.Sequential([
        # 卷积层1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),  
        
        # 卷积层2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),  
        
        # 卷积层3
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),  
        
        # 展平层
        tf.keras.layers.Flatten(),
        
        # 全连接层
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),   
        
        # 输出层
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# 3. 简单可视化模块
def visualize_training_results(history):
    # 创建图形
    plt.figure(figsize=(12, 4))
    
    # 1. 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('model accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    
    # 2. 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('model loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 训练统计信息
    print("\n训练统计:")
    print(f"最终训练准确率: {history.history['accuracy'][-1]:.4f}")
    print(f"最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")
    print(f"最终训练损失: {history.history['loss'][-1]:.4f}")
    print(f"最终验证损失: {history.history['val_loss'][-1]:.4f}")

# 4. 可视化预测结果
def visualize_predictions(model, x_test, y_test, num_samples=10):
    predictions = model.predict(x_test[:num_samples])
    
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        
        true_label = np.argmax(y_test[i])
        pred_label = np.argmax(predictions[i])
        confidence = np.max(predictions[i])
        
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f'T: {true_label}\nP: {pred_label}\nC: {confidence:.2f}', 
                  color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 5. 主程序
def main():
    # 加载数据
    print("加载数据...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    print(f"训练集形状: {x_train.shape}")
    print(f"测试集形状: {x_test.shape}")
    # 划分验证集
    x_train_final, x_val, y_train_final, y_val = train_test_split(
        x_train, y_train, 
        test_size=0.125, 
        random_state=42
    )
    
    # 创建模型
    print("创建模型...")
    model = create_model()
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 显示模型结构
    model.summary()
    
    # 训练模型
    print("开始训练...")
    history = model.fit(
        x_train_final, y_train_final,
        batch_size=64,
        epochs=20,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    # 评估模型
    print("评估模型...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 可视化训练结果
    visualize_training_results(history)
    
    # 可视化预测结果
    visualize_predictions(model, x_test, y_test)
    
    # 保存模型
    model.save('mnist_cnn_model.h5')
    print("模型已保存为 'mnist_cnn_model.h5'")
    
    # 返回训练好的模型和评估结果
    return model, test_accuracy

# 执行主程序
if __name__ == "__main__":
    # 训练模型
    print("=" * 50)
    print("训练模型")
    print("=" * 50)
    model, accuracy = main()