import matplotlib.pyplot as plt

models = ['0~5 m', '5~10 m', '10~15 m', '15~20 m', '20~25 m', '25~30 m', '30m up']
org_640_192 = [0.63, 0.50, 1.55, 1.58, 3.08, 11.09, 13.73]
org_1024_320 = [0.23, 0.51, 1.62, 1.62, 3.66, 12.05, 13.09]
gt_640x192 = [0.40, 0.43, 1.46, 1.94, 2.97, 13.52, 14.62]



x = np.arange(len(models))
width = 0.3
plt.rcParams["figure.figsize"] = (20,10)
plt.bar(x, org_640_192, width, color='green', label='org_640_192')
plt.bar(x + width, org_1024_320, width, color='blue', label='org_1024_320')
plt.bar(x + 2 * width, gt_640x192, width, color='red', label='gt_pose_640x192')
plt.xticks(x + width, models)
plt.ylabel('RMSE')
plt.title('Depth estimation RMSE')
plt.legend(bbox_to_anchor=(1,1), loc='upper left')
plt.show()