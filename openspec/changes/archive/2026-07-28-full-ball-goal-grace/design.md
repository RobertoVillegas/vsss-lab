## Decisions

1. A positive goal occurs when ball center x exceeds
   `field_length / 2 + ball_radius` inside the mouth; negative is symmetric.
2. Detect threshold crossing from previous to current physics state so a ball
   remaining in the goal cannot score repeatedly.
3. Accumulate events across native repeated steps so the Python control frame
   cannot lose a short-lived crossing event.
4. Reward the goal on the event frame, continue control for 50 frames at 20 ms,
   and terminate/reset only after that one-second closure.
5. Keep exact canvas geometry; fix measured physical overlap in Rapier rather
   than hiding it in the viewer.
