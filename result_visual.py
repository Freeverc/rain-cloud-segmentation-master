import cv2
import numpy as np

transparent = 0.5
src_img = '5.png'
pred_img = '5_pred.png'
src_image = cv2.cvtColor(cv2.imread(src_img,-1), cv2.COLOR_BGR2RGB)
pred_image = cv2.cvtColor(cv2.imread(pred_img,-1), cv2.COLOR_BGR2RGB)
result_image = np.array(src_image[:])
arg = np.argwhere(pred_image[:, :, 0] == 250)
mat = transparent * np.array(src_image[arg[:, 0], arg[:,1],:])
print(mat.shape)
result_image[arg[:, 0], arg[:,1],:] = mat + [255 *(1 - transparent),0,0]
cv2.imwrite("%s_visual.png"%('5'),cv2.cvtColor(np.uint8(result_image), cv2.COLOR_RGB2BGR))