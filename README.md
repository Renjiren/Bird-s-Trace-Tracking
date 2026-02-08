# Bird-s-Trace-Tracking

## Abastract

Tracking fast-moving objects in unconstrained natural videos remains a challenging problem due to background dynamics, illumination changes, camera motion, and scale variation. Classical motion-based pipelines offer efficiency and interpretability, whereas modern deep-learning-based trackers achieve high performance at greater computational cost. This work revisits classical computer vision approaches for multi-object tracking in the context of bird detection and tracking by systematically designing and evaluating an OpenCV-based pipeline against a state-of-the-art deep tracker, NetTrack, on the BFT dataset. 

![Birds' Track Architecture](readme_data/pic_framework.png)

The proposed pipeline incorporates camera motion compensation, adaptive motion-based detection, region refinement, and Kalman-filter-based data association. Quantitative experiments using MOTA, IDF1,IDs, NP and FP demonstrate that the classical pipeline achieves competitive results in simpler scenarios（e.g., dataset Su2001), while deep-learning-based methods consistently outperform it in highly dynamic and cluttered environments(e.g.,dataset Ci3001).These findings reveal a clear trade-off between tracking
accuracy and computational efficiency.

![e.g.,Su2001](readme_data/Su2001.jpg)![e.g.,Ci3001](readme_data/Ci3001.jpg)

The dataset is available at https://drive.google.com/drive/folders/140mPnOVZY - 2apH76at9yYuVGIDWOvsH _ ?usp =share_link(Due to size limitations, the dataset is not included in this repository).


## Evaluation Results

We compare our classical computer-vision-based tracking pipeline with **NetTrack**, a
state-of-the-art deep learning tracker, on the BFT validation set.

### Quantitative Comparison 

(NetTrack VS CV-based model)

![NetTrack Evaluation](readme_data/nettrack_new_quant.jpg)
![CV-based model Evaluation](readme_data/cv_new_quant.jpg)

The table above shows the official NetTrack evaluation results reported using standard
MOT metrics (MOTA, IDF1, FP, FN).
NetTrack achieves strong quantitative performance on most sequences, benefiting from
deep object detection and point-level feature tracking.

Our results can be seen in readme_data folder.

### Runtime Analysis

![NetTrack Runtime](readme_data/nettrack_time.jpg)
![CV Runtime](readme_data/cv_new_time.jpg)

The figures above compare the per-video runtime of NetTrack and our proposed classical
CV-based pipeline. (30 hours 41 mins [NetTrack] VS 1 hour 51 min [OurModel])
While NetTrack relies on deep neural network inference and GPU acceleration,
our method runs entirely on CPU using lightweight OpenCV operations.

Overall, our pipeline is **significantly faster** than NetTrack on long sequences,
demonstrating a clear trade-off between tracking accuracy and computational efficiency.
