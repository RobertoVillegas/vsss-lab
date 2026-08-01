# Tasks

- [x] Accept ADR 0018 before implementation begins.
- [x] Restart play at the quadrant free-ball mark after the rule's impasse interval.
- [x] Move a robot inside the clearance to its own half.
- [x] Leave a goal-area impasse unrepositioned, since a goal kick is not modelled.
- [x] Count free balls per world so the impasse rate stays observable.
- [x] Retire the stagnation terminal, keeping its configuration keys loadable.
- [x] Match the rulebook wall thickness of 2.5 cm.
- [x] Add a test that an impasse restarts instead of ending the episode.
- [x] Complete local gates: full suite, `mise run lint`, OpenSpec strict, end-to-end smoke.
- [ ] Model the goal kick of rule 14, including attacking with more than one robot in the
      opponent's goal area.
- [ ] Model the defensive penalty of rule 9.3.
- [ ] Model ball retention and the thirty per cent covering limit of rules 9.1 and 17. The
      longest captured episode of an earlier run held the ball for twenty-nine seconds, which
      those rules would call against us, so the simulation may reward an illegal behaviour.
- [ ] Define the goalkeeper by dwell time inside its own area, with the ten-second clearance
      obligation of rule 3.2.3, instead of by geometry alone.
- [ ] Confirm the lateral free-ball coordinate against the rulebook figure; the horizontal
      37.5 cm is stated, the lateral placement is read as the quadrant centre.
