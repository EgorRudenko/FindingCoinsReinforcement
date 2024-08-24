# Finding Coins Reinforcement
 Das ist ein Projekt für die Schule
 Die Aufgabe kann durch effektivere Methoden sowohl aus Machine Learning als auch aus anderen Branchen gelöst werden.
 Also das einzige Ziel von diesem Projekt ist einfach ein Versuch KI zu benutzen

 # Nutzen
 Die nötige Python Module sind websockets, numpy und scipy. Die alle sind durch ```pip install -r requirements.txt``` runterlädtbar
 Man braucht auch [godot](https://godotengine.org/) engine</br>
 Erst soll man main.py (**nicht biggerMain.py**, weil das nicht trainiert ist) öffnen, dann godot projekt starten</br>

 Die Variablen, die man ändern wollen könnte:</br>
 Area2D.gd: </br>
 timeMultiplier -> wie schnell die Zeit verläuft. Groß für lernen, irgendwie für Nutzen</br>
 maxLifeWithoutMoney  -> Wie viele Sekunden lebt Agent ohne Münzen</br>
 forceApplied -> Wie schnell bewegt sich der Agent</br>

 main.py</br>
 toLearn</br>
 toLoad</br>
 # Quellen
 Die klenere Quellen sind in der Code als Kommentare geschrieben</br>
 Ich habe mein andere Projekt zum Vergleichen genutzt [cart pole](https://github.com/EgorRudenko/CartPoleDeepReinforcementLearning), welche auf einen [anderen](https://github.com/numpy/numpy-tutorials/blob/main/content/tutorial-deep-reinforcement-learning-with-pong-from-pixels.md) basiert ist
# English 
 School project
 The task is probably most efficiantly solved by other methods, but [+AI](https://www.reddit.com/r/mathmemes/comments/1el7jy2/since_too_many_people_are_asking_heres_the/)
 
 # To run 
 One needs websockets, numpy and scipy modules in python they are also can be found in requirements.txt</br>
 ```pip install -r requirements.txt```</br>
 First run main.py script from python folder (**not biggerMain.py**, because it's not trained) 
 Then [godot](https://godotengine.org/) project</br>

 some variables one may want to change:</br>
 Area2D.gd: </br>
 timeMultiplier -> it changes how fast time goes. Set big  for learning and anyhow for using</br>
 maxLifeWithoutMoney  -> how often agent must grab coins to live</br>
 forceApplied -> changes how fast agent can move</br>

 main.py</br>
 toLearn</br>
 toLoad</br>

 # Sources
 The minor sources of information and code are mentioned in comments in code</br>
 I used a few times my other project for reference: [cart pole](https://github.com/EgorRudenko/CartPoleDeepReinforcementLearning), which in its turn is based on [pong](https://github.com/numpy/numpy-tutorials/blob/main/content/tutorial-deep-reinforcement-learning-with-pong-from-pixels.md)
