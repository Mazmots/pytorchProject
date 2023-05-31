# 数据集预处理
"""
数据来源
知乎https://zhuanlan.zhihu.com/p/388676784

现在目录结构：
root='E:\\kagglecatsanddogs\\'
cat = root + 'PetImages\\Cat'
dog = root + 'PetImages\\Dog'

目标结构：
'data\cad\\train\dog'
'data\cad\\train\cat'

'data\cad\\test\dog'
'data\cad\\test\cat'


"""



def pre_deal_data():
    root_path = r'E:\kagglecatsanddogs\PetImages'

    cat_dir = os.path.join(root_path, 'Dog')
    cat_imgs = os.listdir(cat_dir)
    # random_idxs = random.sample(range(len(cat_imgs)), 2500)
    #
    # test_dir = 'data2/test/cat/'
    # if not os.path.exists(test_dir):
    #     os.makedirs(test_dir)
    #
    # for idx in random_idxs:
    #     shutil.move(os.path.join(cat_dir, cat_imgs[idx]), os.path.join(test_dir, cat_imgs[idx]))
    #
    # test_imgs = os.listdir(test_dir)
    # print(len(test_imgs))
    train_dir = 'catdog/train/dog'
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    files = os.listdir(cat_dir)
    print(files)
    for file_name in files:
        shutil.move(os.path.join(cat_dir, file_name), os.path.join(train_dir, file_name))

    pass




def deal_data():
    cat_tgt = r'E:\catdog\train\cat'
    dog_tgt = r'E:\catdog\train\dog'

    # 图片尺寸全部变为100*100
    cat_train_dir = r'E:\cat_dog2\train\cat'
    dog_train_dir = r'E:\cat_dog2\train\dog'
    cat_img_list = os.listdir(cat_train_dir)
    for img_name in cat_img_list:
        print(f'{img_name} reading')
        img = cv2.imread(os.path.join(cat_train_dir, img_name), cv2.IMREAD_COLOR)
        print(f'{img_name} resizing')
        try:
            tmp = cv2.resize(img, (50, 50), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        except:
            continue
        print(f'{img_name} saving')
        cv2.imwrite(os.path.join(cat_tgt, img_name), tmp)
        print(f'{img_name} save finish')

    dog_img_list = os.listdir(dog_train_dir)
    for img_name in dog_img_list:
        print(f'{img_name} reading')
        img = cv2.imread(os.path.join(dog_train_dir, img_name), cv2.IMREAD_COLOR)
        print(f'{img_name} resizing')
        try:
            tmp = cv2.resize(img, (50, 50), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        except:
            continue
        print(f'{img_name} saving')
        cv2.imwrite(os.path.join(dog_tgt, img_name), tmp)
        print(f'{img_name} save finish')