# A stochastic optimization based stochastic optimal control framework for artifical pancreas (NYU Courant SURE 2019)
------------
This repository features my joint research project with Xinyu Li on a stochastic optimization based stochastic optimal control framework for artifical pancreas in my junior year at NYU Courant Institute (advised by Prof. Jonathan Goodman, through NYU Courant Summer Undergraduate Research Experience program). In this project, we formulated a model of the insulin-glucose metabolism to characterize the dynamics of insulin and glucose levels, as well as the impact of meal intake and insulin injection on them, in a hypothetical type 1 diabetes patient. To prepare the system for real-world applications, we denoised the measurements using Kalman filter, and determined the optimal insulin injection amount for the patient by modern stochastic optimization algorithms from deep learning. 

References:
* Kalman filter: Ribeiro, Maria Isabel. "Kalman and extended kalman filters: Concept, derivation and properties." Institute for Systems and Robotics 43.46 (2004): 3736-3741.
* The minimal model for the insulin-glucose metabolism: 
* - Ni, Ta-Chen, Marilyn Ader, and Richard N. Bergman. "Reassessment of glucose effectiveness and insulin sensitivity from minimal model analysis: a theoretical evaluation of the single-compartment glucose distribution assumption." Diabetes 46.11 (1997): 1813-1821.
* - Cobelli, C. L. A. U. D. I. O., et al. "Estimation of insulin sensitivity and glucose clearance from minimal model: new insights from labeled IVGTT." American Journal of Physiology-Endocrinology And Metabolism 250.5 (1986): E591-E598.
* - Natalucci, Silvia, et al. "Insulin sensitivity and glucose effectiveness estimated by the minimal model technique in spontaneously hypertensive and normal rats." Experimental Physiology 85.6 (2000): 777-781.
* - Ludwig, Tomas, and Ivan Ottinger. "Identification of T1DM minimal model using non-consistent data from IVGTT." Journal of Electrical Systems and Information Technology 1.2 (2014): 144-149.
* - Winkel, Brian. "2017-Gupta, Richa and Deepak Kumar-Numerical Model for Glucose Metabolism for Various Types of Food and Effect of Physical Activities on Type 1 Diabetic Patient." (2020).
* - Ludwig, T., et al. "TYPE 1 DIABETES MELLITUS MODEL: SIMULATION STUDY."
* - Calm, Remei, et al. "Prediction of glucose excursions under uncertain parameters and food intake in intensive insulin therapy for type 1 diabetes mellitus." 2007 29th Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE, 2007.
* - Nyman, Elin, Gunnar Cedersund, and Peter Strålfors. "Insulin signaling–mathematical modeling comes of age." Trends in Endocrinology & Metabolism 23.3 (2012): 107-115.
* Stochastic optimization:
* - Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods for online learning and stochastic optimization." Journal of machine learning research 12.7 (2011).
* - Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
* - Hinton, Geoffrey, Nitish Srivastava, and Kevin Swersky. "Neural networks for machine learning lecture 6a overview of mini-batch gradient descent." Cited on 14.8 (2012): 2.



<!--- To clone this repository to your own workspace (e.g. your laptop), run the command

    git clone https://github.com/silvialuu/sure-repo
    
in your terminal. I suggest that you could first cd to the directory where you want to put this repository, and then run the clone command. Otherwise, this repository will be cloned to the folder Macintosh HD -> Users -> YOUR_USER_NAME by default. 

To keep the repository on your own workspace up to date with the changes that have been made to the remote location by your collaborators, run the command

	git pull

I strongly suggest that you run this command each time before you want to make any change in your local repository. In particular, if you 'push' to the remote location before you 'pull' the changes that have been made there by your collaborators, you will run into trouble. 

To upload the changes you have made on your own computer, run commands

	git add FILENAME1 FILENAME2 FILENAME3
	git commit -m "SOMETHING USEFUL"
	git push origin master

The first step `git add ...` allows you to specify the list of files that you want to add to the repository and then send to the remote location. 

The second step `git commit - m "...."` adds the files to the repository (but only in your workspace, not in the remote locations). SOMETHING USEFUL should describe the reason for the commit (e.g.  "add a new function F"). It tells you and your collaborators what you did. 

The final step `git push origin master` tells `git` to transmit the files from your workspace to the remote location on GitHub. "master" may get replaced by different branch names if later we decide to create and use new branches. --->



