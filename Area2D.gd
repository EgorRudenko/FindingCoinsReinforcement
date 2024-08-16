extends Area2D

var speed = 400;
var visionDist = 150;
var timerTillDeath = 0;
var maxLifeWithoutMoney = 5;
var gameOver = false
var action = Vector2(0,0)

# Fremdquellecode. Quelle: https://docs.godotengine.org/en/stable/classes/class_websocketpeer.html (Primär)
# bzw. https://github.com/EgorRudenko/CartPoleDeepReinforcementLearning (Daraus kopierte ich eigentlich, weil ich das schon etwas für meine Bedürfnisse umgeschrieben habe)

var client = WebSocketPeer.new();
var url = "ws://localhost:5000";
# Ende Fremdquellecode

func move(delta, move):
	if abs(position[0]) > 530:
		gameOver = true
	if abs(position[1]) > 300:
		gameOver = true
	position += move * delta * speed

func divArray(a, b):
	for i in range(len(a)):
		a[i] /= b
	return a

func readActionFromKeyboard():
	var move = Vector2(0,0)
	if Input.is_action_pressed("bottom"): 
		move += Vector2(0, 1)
	if Input.is_action_pressed("top"): 
		move += Vector2(0, -1)
	if Input.is_action_pressed("right"): 
		move += Vector2(1, 0)
	if Input.is_action_pressed("left"): 
		move += Vector2(-1, 0)
	return move

func check_on_outputs():
	var diff = get_parent().get_node("Area2D2").position - position
	var dist = diff.length()
	var visionRays = [0,0,0,0,0]
	#print(dist)
	if dist < visionDist:
		var angle = atan(diff[1]/diff[0])
		if diff[0] < 0 and diff[1] < 0:
			angle -= PI
		elif diff[0] < 0 and diff[1] > 0:
			angle +=  PI
		angle += PI
		# vision rays : 2PI/5 * k, where k is from 0 to 4 (k element N)
		for i in range(len(visionRays)):
			visionRays[i] = cos(i*(2*PI/5) - angle) * dist
	return divArray(visionRays,visionDist) 

func reinit():
	timerTillDeath = 0;
	position = Vector2(0,0)
	gameOver = false

func communicate():
	# Fremdquellecode (mit einigen Änderungen). Quelle: https://docs.godotengine.org/en/stable/classes/class_websocketpeer.html
	# bzw. https://github.com/EgorRudenko/CartPoleDeepReinforcementLearning
	client.poll()
	var state = client.get_ready_state()
	if state == WebSocketPeer.STATE_OPEN:
		var rays = check_on_outputs()
		client.send_text(str(position[0]/550, " ", position[1]/300, " ", rays[0], " ", rays[1], " ", rays[2], " ", rays[3], " ", rays[4]))
		action = Vector2(0,0);
		while client.get_available_packet_count():
			var temp = Array(client.get_packet().get_string_from_utf8().split(" ")).map(func(element): int(element))
			action = Vector2(temp[0], temp[1])
			# Why can't I just construct Vector from an array? And why is vector max 4 elements big? It starts to annoy me
	elif state == WebSocketPeer.STATE_CLOSED:
		set_process(false)
	# Ende Fremdquellecode

func _process(delta):
	move(delta, readActionFromKeyboard())
	timerTillDeath += delta
	if timerTillDeath >= maxLifeWithoutMoney:
		gameOver = true
	if gameOver == true:
		reinit()


func _on_area_entered(area):
	timerTillDeath = 0


