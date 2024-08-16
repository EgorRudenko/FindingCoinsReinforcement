extends Area2D

var rng = RandomNumberGenerator.new()

func _on_area_entered(area):
	while (get_parent().get_node("Area2D").position - position).length() < 60:
		# it doesn't detect entering if it "spawned" in the player, so this is the best I could think of
		position = Vector2(rng.randi_range(-550,550), rng.randi_range(-300, 300))
