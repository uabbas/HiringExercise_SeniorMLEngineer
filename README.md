# Center for Medical Innovation, Software, and Technology
### Sidra Medical Research Center
![](https://raw.githubusercontent.com/CMIST/HiringExercise_MLCVEngineer/master/logo_cmist.png "CMIST LOGO")

Dear Candidate,

Congratulations on your candidacy to the CMIST Division (Center for Medical Innovation, Software, and Technology) at Sidra Medical Research Center.  Our hiring process is gated and involves four steps:
1.	**Technical Competency** - evaluated from job application
2.	**Technical Mastery** - evaluated via application and phone screen with hiring manager
3.	**Delivery Capability** - evaluated via this take-home exercise
4.	**Organizational Fit** - evaluated via group interview 

This repository details item three – a take-home exercise which aims to evaluate your ability to deliver.  With this, we are observing three major things:
1.	Scientific and Technical Correctness
2.	Communication and Presentation
3.	Self-Management

### The Exercise

Please create a model which can be used to recognize and annotate cats in photos.  You are welcome to use any training set available online to train the model.  Please deliver two scripts: one to train your model and one which applies the model onto our test set of images (test_set.zip) and draws bounding boxes around cats on the test set as such:

![](https://raw.githubusercontent.com/CMIST/HiringExercise_MLCVEngineer/master/catbb.png "Alf with Lucky")

Please deliver back these two scripts (and everything else referenced) as well as a corresponding set of images with the bounding boxes.  *Do not* submit any large training sets, rather include scripts to acquire them before starting the training.  Your scripts should be all-encompassing, including pipeline steps and acquisition of any training sets.  If you need a setup script, that can also be included.

Your submission should be via Github.  You can make your repo public or private (in which case, add https://github.com/CMIST as a collaborator.)  Answers to the discussions questions should also be delivered via the repository as either Markdown, Plaintext, Jupyter Notebook, or PDF.

The exercise can be done in any language (though we prefer python) as long as the submission can be successfully executed.  Please do not use paid platforms like MATLAB.  If you are completely agnostic, you can use our in-house standard: TensorFlow + Keras or Caffe2.  We plan to execute this on an AWS EC2 G-series machine.  To this point, please detail your selected environment, machine learning / computer vision platform and any other setup steps so we can re-train and re-test your model from scratch.

There is no need to re-create the wheel here, use any and all open-source or vendor-provided scripts that you can.  Ensure to include them so the submission is all-encompassing.  **Do not** include anything irrelevant to the goal of this exercise, we want the minimal set of work which achieves the goal and nothing more.

### Discussion Questions
1.	What network/model did you select, and why
2.	Describe your pipeline and pre-processing steps
3.	What steps did you take to get the best accuracy
4.	How long did your training and inference take, how could you make these faster?
5.	If you had more time, how would you expand on this submission?

### Rationale for this Exercise
Without working with confidential medical images, this exercise closely resembles types of work we do in the group, specifically:
* Dealing with heterogeneous in-the-wild samples (various sizes, various aspect ratios.)  The same pre-processing script (using “tiling” which I had hinted at before) should be able to handle everything.
* Dealing with 0,1,1+ cases per image
* Dealing with multiple file formats (easily handled via PIL)
* Communicating ideas
* Managing own efforts 

Good Luck!

