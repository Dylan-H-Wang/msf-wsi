val_f1_micros = []
val_f1_macros = []
tumor_f1s = []
stroma_f1s = []
infla_f1s = []
necr_f1s = []
other_f1s = []
val_acc_micros = []
val_acc_macros = []

for i in range(len(val_dataset)):
    idx = i
    filename = val_dataset.files[idx]
    df = val_dataset.data_df[val_dataset.data_df['filename']==filename].reset_index(drop=True)
    a_trans = albu.CenterCrop(256, 256, always_apply=True)
    all_imgs = []
    for img_name in df["filename_img"]:
        img_path = os.path.join(args.data_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        all_imgs.append(a_trans(image=img)['image'])
    all_imgs = np.stack(all_imgs, axis=0)


    (
        val_f1_micro,
        val_f1_macro,
        tumor_f1_,
        stroma_f1_,
        infla_f1_,
        necr_f1_,
        other_f1_,
        val_acc_micro,
        val_acc_macro,
    ) = validate(val_dataset[idx], all_imgs, model, args)

    val_f1_micros.append(val_f1_micro)
    val_f1_macros.append(val_f1_macro)
    tumor_f1s.append(tumor_f1_)
    stroma_f1s.append(stroma_f1_)
    infla_f1s.append(infla_f1_)
    necr_f1s.append(necr_f1_)
    other_f1s.append(other_f1_)
    val_acc_micros.append(val_acc_micro)
    val_acc_macros.append(val_acc_macro)

print(
    "=======\n"
    f"All F1 micro: {val_f1_micros:.4f}, macro: {val_f1_macros:.4f}\n"
    f"Tumor F1: {tumor_f1s:.4f}, Stroma F1: {stroma_f1s:.4f}, Infla F1: {infla_f1s:.4f}, "
    f"Necr F1: {necr_f1s:.4f}, Other F1: {other_f1s:.4f}\n"
    f"Accuracy micro: {val_acc_micros:.4f}, macro: {val_acc_macros:.4f}\n"
    "======="
)

