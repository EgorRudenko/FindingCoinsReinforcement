extends Area2D

var rng = RandomNumberGenerator.new()
# These are a bit smaller than window, so it doesnt have to do too much learning at not hitting walls
var xRange = 530;
var yRange = 280;

func randPos():
	position = Vector2(rng.randi_range(-xRange,xRange), rng.randi_range(-280, 280))

func _on_area_entered(area):
	while (get_parent().get_node("Area2D").position - position).length() < 60:
		# it doesn't detect entering if it "spawned" in the player, so this is the best I could think of
		randPos()


func _on_area_2d_game_over():
	randPos()
