# Preserve low-speed ball continuity

Replay `vsss-semantic-run-0005/iteration-000175` exposed a physically invalid
transition: an unobstructed ball moving at 0.1935 m/s became exactly stationary
in one 20 ms control frame. Rapier's generic sleep threshold is 0.4
length-units/s for two seconds, which is too high for a metre-scale VSSS field.

Keep the single ball awake so configured damping and contacts remain the only
causes of velocity change. Robots retain sleeping behavior. This invalidates
training runs produced before the fix for promotion purposes.
