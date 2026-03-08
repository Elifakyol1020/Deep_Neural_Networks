import os
import cv2
import numpy as np


def prepare_training_set(train_folder, max_images_per_class=1000):
    image_vectors = []
    image_labels = []
    label_names = []

    print("Eğitim verileri hazırlanıyor...")

    folders = sorted(os.listdir(train_folder))

    for label_index, folder_name in enumerate(folders):
        folder_path = os.path.join(train_folder, folder_name)

        if not os.path.isdir(folder_path):
            continue

        label_names.append(folder_name)
        loaded_image_count = 0

        file_names = sorted(os.listdir(folder_path))

        for file_name in file_names:
            if loaded_image_count >= max_images_per_class:
                break

            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path)

            if image is None:
                continue

            image = cv2.resize(image, (32, 32))
            image = image.flatten()

            image_vectors.append(image)
            image_labels.append(label_index)
            loaded_image_count += 1

        print(f"{folder_name} klasöründen {loaded_image_count} adet görüntü yüklendi.")

    image_vectors = np.array(image_vectors, dtype=np.float32) / 255.0
    image_labels = np.array(image_labels)

    print("Toplam eğitim verisi:", len(image_vectors))

    return image_vectors, image_labels, label_names


def calculate_l1(vector1, vector2):
    total_distance = 0.0

    for i in range(len(vector1)):
        total_distance += abs(vector1[i] - vector2[i])

    return total_distance


def calculate_l2(vector1, vector2):
    total_distance = 0.0

    for i in range(len(vector1)):
        difference = vector1[i] - vector2[i]
        total_distance += difference * difference

    return np.sqrt(total_distance)


def predict_class(train_data, train_labels, sample_image, k_value, distance_choice):
    distance_list = []

    for i in range(len(train_data)):
        if distance_choice == "L1":
            distance = calculate_l1(train_data[i], sample_image)
        else:
            distance = calculate_l2(train_data[i], sample_image)

        distance_list.append((distance, train_labels[i]))

    distance_list.sort(key=lambda item: item[0])

    nearest_neighbors = distance_list[:k_value]

    vote_table = {}

    for _, class_label in nearest_neighbors:
        if class_label in vote_table:
            vote_table[class_label] += 1
        else:
            vote_table[class_label] = 1

    predicted_label = max(vote_table, key=vote_table.get)
    return predicted_label


print("=== CIFAR-10 k-NN Sınıflandırma Uygulaması ===")

train_path = "data/cifar10/cifar10/train"

if not os.path.exists(train_path):
    print("Eğitim veri klasörü bulunamadı.")
    print("train_path değişkenini kendi klasör yapına göre güncelle.")
    raise SystemExit

X_train, y_train, class_names = prepare_training_set(train_path, max_images_per_class=1000)

print("\nMesafe türünü seçiniz:")
print("1 - L1 (Manhattan)")
print("2 - L2 (Euclidean)")

user_choice = input("Seçiminiz: ").strip()

if user_choice == "1":
    selected_metric = "L1"
elif user_choice == "2":
    selected_metric = "L2"
else:
    print("Geçersiz seçim yapıldı. Varsayılan olarak L2 kullanılacak.")
    selected_metric = "L2"

try:
    k_value = int(input("k değerini giriniz: ").strip())
except ValueError:
    print("Lütfen geçerli bir tam sayı giriniz.")
    raise SystemExit

if k_value <= 0:
    print("k değeri 0'dan büyük olmalıdır.")
    raise SystemExit

if k_value > len(X_train):
    print("k değeri eğitim veri sayısından büyük olamaz.")
    raise SystemExit

test_image_path = input("Test edilecek görselin yolunu giriniz: ").strip()

if not os.path.exists(test_image_path):
    print("Girilen görsel yolu bulunamadı.")
    raise SystemExit

test_image = cv2.imread(test_image_path)

if test_image is None:
    print("Görsel okunamadı.")
    raise SystemExit

test_image = cv2.resize(test_image, (32, 32))
test_image = test_image.flatten().astype(np.float32) / 255.0

predicted_index = predict_class(
    train_data=X_train,
    train_labels=y_train,
    sample_image=test_image,
    k_value=k_value,
    distance_choice=selected_metric
)

real_class_name = os.path.basename(os.path.dirname(test_image_path))

print("\n=== Sonuç ===")
print("Seçilen mesafe türü:", selected_metric)
print("k değeri:", k_value)
print("Gerçek sınıf:", real_class_name)
print("Tahmin edilen sınıf:", class_names[predicted_index])
