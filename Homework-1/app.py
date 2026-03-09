import os
import cv2
import numpy as np
import streamlit as st


# Streamlit sayfa ayari.
st.set_page_config(page_title="CIFAR-10 k-NN Arayuzu", layout="wide")


# L1 (Manhattan) uzakligi: mutlak farklarin toplami.
def calculate_l1(vector1, vector2):
    total_distance = 0.0
    for i in range(len(vector1)):
        total_distance += abs(vector1[i] - vector2[i])
    return total_distance


# L2 (Euclidean) uzakligi: kare farklar toplaminin karekoku.
def calculate_l2(vector1, vector2):
    total_distance = 0.0
    for i in range(len(vector1)):
        difference = vector1[i] - vector2[i]
        total_distance += difference * difference
    return np.sqrt(total_distance)


@st.cache_data(show_spinner=False)
def prepare_training_set(train_folder, max_images_per_class=1000):
    # Egitim klasorlerindeki resimleri sayisal vektore cevirir.
    image_vectors = []
    image_labels = []
    label_names = []

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

    image_vectors = np.array(image_vectors, dtype=np.float32) / 255.0
    image_labels = np.array(image_labels)

    return image_vectors, image_labels, label_names


def predict_class(train_data, train_labels, sample_image, k_value, distance_choice):
    # Tum egitim orneklerine uzaklik hesaplanir ve en yakin k komsu oylanir.
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


def get_top_predictions(train_data, train_labels, sample_image, k_value, distance_choice, top_n=5):
    # k komsudan gelen oy dagilimina gore en iyi siniflari siralar.
    distance_list = []

    for i in range(len(train_data)):
        if distance_choice == "L1":
            distance = calculate_l1(train_data[i], sample_image)
        else:
            distance = calculate_l2(train_data[i], sample_image)

        distance_list.append((distance, int(train_labels[i])))

    distance_list.sort(key=lambda item: item[0])
    nearest_neighbors = distance_list[:k_value]

    # Oylar eşitse ortalama uzaklığı düşük olan sınıfı üstte göster.
    vote_table = {}
    for distance, class_label in nearest_neighbors:
        if class_label not in vote_table:
            vote_table[class_label] = {"votes": 0, "distance_sum": 0.0}
        vote_table[class_label]["votes"] += 1
        vote_table[class_label]["distance_sum"] += distance

    ranked = []
    for class_label, values in vote_table.items():
        avg_distance = values["distance_sum"] / values["votes"]
        ranked.append((class_label, values["votes"], avg_distance))

    ranked.sort(key=lambda item: (-item[1], item[2]))
    return ranked[:top_n]


st.title("CIFAR-10 k-NN Siniflandirma")

with st.sidebar:
    # Kullanici bu panelden model parametrelerini girer.
    st.header("Ayarlar")
    train_path = st.text_input("Egitim klasoru yolu", value="data/cifar10/cifar10/train")
    max_images_per_class = st.number_input(
        "Sinif basina maksimum goruntu",
        min_value=1,
        max_value=5000,
        value=1000,
        step=50,
    )
    selected_metric = st.selectbox("Mesafe turu", options=["L1", "L2"], index=1)
    k_value = st.number_input("k degeri", min_value=1, value=3, step=1)

st.subheader("Test Goruntusu")
test_image_path = st.text_input(
    "Test edilecek goruntunun yolu",
    value="data/cifar10/cifar10/train/airplane/0001.png",
)

run_prediction = st.button("Tahmin Et", type="primary")

if run_prediction:
    # Tahmin butonuna basildiginda veri yukleme ve test akisi baslar.
    if not os.path.exists(train_path):
        st.error("Egitim veri klasoru bulunamadi. Yol bilgisini kontrol edin.")
        st.stop()

    with st.spinner("Egitim verileri yukleniyor..."):
        X_train, y_train, class_names = prepare_training_set(
            train_folder=train_path,
            max_images_per_class=int(max_images_per_class),
        )

    if len(X_train) == 0:
        st.error("Egitim verisi yuklenemedi.")
        st.stop()

    if k_value > len(X_train):
        st.error("k degeri egitim veri sayisindan buyuk olamaz.")
        st.stop()

    if not os.path.exists(test_image_path):
        st.error("Test goruntusu yolu bulunamadi.")
        st.stop()

    test_image = cv2.imread(test_image_path)

    if test_image is None:
        st.error("Test goruntusu okunamadi.")
        st.stop()

    resized = cv2.resize(test_image, (32, 32))
    # Test resmi egitimle ayni olcege cekilir (normalize edilir).
    flattened = resized.flatten().astype(np.float32) / 255.0

    predicted_index = predict_class(
        train_data=X_train,
        train_labels=y_train,
        sample_image=flattened,
        k_value=int(k_value),
        distance_choice=selected_metric,
    )

    predicted_class_name = class_names[predicted_index]
    real_class_name = os.path.basename(os.path.dirname(test_image_path))
    top_predictions = get_top_predictions(
        train_data=X_train,
        train_labels=y_train,
        sample_image=flattened,
        k_value=int(k_value),
        distance_choice=selected_metric,
        top_n=5,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB), caption="Test goruntusu", use_container_width=True)

    with col2:
        st.success(f"Tahmin edilen sinif: {predicted_class_name}")
        st.info(f"Gercek sinif: {real_class_name}")
        if predicted_class_name == real_class_name:
            st.write("Durum: Dogru tahmin")
        else:
            st.write("Durum: Yanlis tahmin")

    st.subheader("Ilk 5 Tahmin")
    # Ilk 5 tahmin sinif, oy ve ortalama uzaklik bilgisiyle tabloda gosterilir.
    top_rows = []
    for index, (class_label, votes, avg_distance) in enumerate(top_predictions, start=1):
        top_rows.append(
            {
                "Sira": index,
                "Sinif": class_names[class_label],
                "Oy": votes,
                "Ortalama Uzaklik": round(float(avg_distance), 6),
            }
        )
    st.table(top_rows)
