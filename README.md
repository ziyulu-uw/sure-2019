# sure-repo
------------
To clone this repository to your own workspace (e.g. your laptop), run the command
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

The final step `git push origin master` tells `git` to transmit the files from your workspace to the remote location on GitHub. "master" may get replaced by different branch names if later we decide to create and use new branches. 



