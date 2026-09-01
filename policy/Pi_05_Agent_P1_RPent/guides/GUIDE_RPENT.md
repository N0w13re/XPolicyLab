# RoboDojo RPent Registered-Tool Guide

## Geometry

Start from `view_env_state`. Grounding returns a bounding box tied to one
immutable RGB-D state. Use `query_world_map` or several `sample_world_xyz`
calls on that exact step. World pixels use `[row,col]`; poses use
`[x,y,z,qw,qx,qy,qz]` in metres in the local world frame.

An object surface point is not an end-effector target. Add clearance and the
task-appropriate EEF/TCP offset. Use `move_to` only for free-space transport,
staging, retreat, or a small verified correction. Preserve orientation and
gripper state unless changing them is intentional.

## VLA and State

Every `pi05_act` sends the complete original task instruction to Pi_05. Its
`focus` only records the current phase. Execute short prefixes and inspect fresh
views after each call.

A closed gripper is not a grasp. Verify that the selected object leaves its
source and moves with the TCP. A release is valid only after a verified hold and
a successful carrying move. For handover, verify the receiver before opening
the giver.

## Recovery and Finish

After failure, classify the first unmet gate: perception, planning/collision,
grasp, lost hold, transport, release, placement, or final relation. Do not
repeat an unchanged failed target twice. Change one meaningful variable, switch
to Pi_05 for contact-rich motion, replan, or safely abort.

Native step limits are absolute. Preserve turns for final observation and
`finish`. Only fresh official environment success proves task completion.
