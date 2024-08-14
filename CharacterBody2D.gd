extends CharacterBody2D

var speed = 400;

func _ready():
	pass
func move(delta):
	var move = Vector2(0,0)
	if Input.is_action_pressed("bottom"): 
		move += Vector2(0, speed*delta)
	if Input.is_action_pressed("top"): 
		move += Vector2(0, -speed*delta)
	if Input.is_action_pressed("right"): 
		move += Vector2(speed*delta, 0)
	if Input.is_action_pressed("left"): 
		move += Vector2(-speed*delta, 0)
	position += move
func _process(delta):
	move(delta)
