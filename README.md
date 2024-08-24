# Finding Coins Reinforcement
 Das ist ein Projekt für die Schule
 Die Aufgabe kann durch Effektivere Methoden sowohl aus Machine Learning als auch aus anderen Branchen gelöst.
 Also das einzige Ziel davon ist einfach KI zu versuchen

 # Nutzen
 Die nötige Python Module sind websockets, numpy und scipy. Die alle sind durch ´´´pip install -r requirements.txt´´´ runterlädtbar
 Man braucht auch [godot](https://godotengine.org/) engine

 Die Variablen, die man ändern wollen könnte:
 Area2D.gd: 
 timeMultiplier -> wie schnell die Zeit verläuft. Groß für lernen, irgendwie für Nutzen
 maxLifeWithoutMoney  -> Wie viele Sekunden lebt Agent ohne Münzen
 forceApplied -> Wie schnell bewegt sich der Agent

 main.py
 toLearn
 toLoad
 # Quellen
 Die klenere Quellen sind in der Code als Kommentare geschrieben
 Ich habe mein andere Projekt zum vergleichen genutzt [cart pole](https://github.com/EgorRudenko/CartPoleDeepReinforcementLearning), welche auf einen [anderen](https://github.com/numpy/numpy-tutorials/blob/main/content/tutorial-deep-reinforcement-learning-with-pong-from-pixels.md) basiert ist
# English 
 School project
 The task is probably most efficiantly solved by other methods, but [+AI](https://www.reddit.com/r/mathmemes/comments/1el7jy2/since_too_many_people_are_asking_heres_the/)
 
 # To run 
 One needs websockets, numpy and scipy modules in python they are also can be found in requirements.txt
 ```pip install -r requirements.txt```
 First run main.py script from python folder (**not biggerMain.py**, because it's not trained) 
 Then [godot](https://godotengine.org/) project

 some variables one may want to change:
 Area2D.gd: 
 timeMultiplier -> it changes how fast time goes. Set big  for learning and anyhow for using
 maxLifeWithoutMoney  -> how often agent must grab coins to live
 forceApplied -> changes how fast agent can move

 main.py
 toLearn
 toLoad

 # Sources
 The minor sources of information and code are mentioned in comments in code
 I used a few times my other project for reference: [cart pole](https://github.com/EgorRudenko/CartPoleDeepReinforcementLearning), which in its turn is based on [pong](https://github.com/numpy/numpy-tutorials/blob/main/content/tutorial-deep-reinforcement-learning-with-pong-from-pixels.md)
