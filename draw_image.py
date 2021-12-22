import matplotlib.pyplot as plt
import cv2
def show(imgs, n, path):
    labels = [['original_image', 'pred_disp'], ['gt_depth', 'car_mask'], ['mask']]

    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=2, nrows=3, squeeze=False, figsize=(32,24))
    for i in range(2):
        for j in range(2):
            #img = img.detach()
            #img = F.to_pil_image(img)
            if i == 1 and j == 0:
                ret, tmp = cv2.threshold(imgs[i * 2 + j], 1e-3, 255, cv2.THRESH_BINARY)
                axs[i, j].imshow(tmp, vmin=0, vmax=255)
                # axs[i, j].imshow(imgs[i * 2 + j], cmap='gray', vmin=0., vmax=80.)
                axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                axs[i, j].set_title(label=labels[i][j])
            else:
                axs[i, j].imshow(imgs[i * 2 + j])
                axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                axs[i, j].set_title(label=labels[i][j])

    axs[2, 0].imshow(imgs[4])
    axs[2, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axs[2, 0].set_title(label=labels[2][0])

    fix.savefig(path + '/%d.png'%(n))