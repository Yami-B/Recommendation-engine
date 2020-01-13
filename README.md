# Recommendation-engine

In this repository, I will show my work on a project about designing a recommendation engine. In this project, a website, storing Machine Learning problems for users to solve, is looking for a way to suggest each user problems that might interest him or her. This is in order to make users progress and keep them interested in using this website. 

For this purpose, the website has delivered 3 csv files enclosing data about users behaviour and problems:
- first file with users ID and the corresponding problems ID on which they worked, along with their number of attempts (ranging from 1 to 6)
- second file focusing on users: Country of residence, registration time, total time spent on the website, rank (expert, advanced, intermediate, beginner), number of points, etc..
- third file focused on the problems: level (A to N, N being the most dfficult), points earned if solved and tags (field on which the problem deals with. Ex: greedy, brute force, implementation, graphes,..)

Finally, the website gave a fourth csv file containing new users, problems associated and from that, we had to predict the number of attempts each of these users we'll make.

# MODUS OPERANDI

My first basic idea was to cluster users depending on their informations (second csv file). So when we have to predict how many attempts user A will make on problem B, we take users from the cluster to which user A belongs, select the users who also worked on problem B and average their attempts on it.
This approach reached a F1 score of 0.47 (and the best score of the competition was around 0.5).

Another approach I tried was complementary to the first one: form clusters of users and identify similar problems, according to their level, the number of points and the tags to also cluster them, and finally average the attempts computed by the users cluster and by the problems cluster. However, the third file containing data about problems was missing a lot of information so I had to find ways to impute values to levels, tags and points (sometimes all of the three). 

