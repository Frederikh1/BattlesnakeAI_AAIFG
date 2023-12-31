#!/bin/bash

#Executable permissions:
chmod +x ../BattlesnakeLinux/battlesnake

win=0
draw=0
lose=0
loopcount=100

# Loop through the game plays
while [ $loopcount -gt 0 ]; do
    ../BattlesnakeLinux/battlesnake play -W 11 -H 11 --name AFK --url https://battlesnakeaiaaifg.kasperdelaxson.repl.co/ --name Adam --url https://battlesnake.up.railway.app/ 1> out.txt 2>&1
    # Battlesnake that uses all 4 snakes, using Adam, and 2 snakes from Coreyja. 
    # ../BattlesnakeLinux/battlesnake play -W 11 -H 11 --name AFK --url https://battlesnakeaiaaifg.kasperdelaxson.repl.co/ --name Adam --url https://battlesnake.up.railway.app/ --name CoreyjaHoveringHobs --url https://battlesnake-rs.fly.dev/hovering-hobbs --name CoreyjaDeviousDevin --url https://battlesnake-rs.fly.dev/devious-devin 1> out.txt 2>&1


    # Checking game outcomes and updating counters
    if grep -q "AFK was the winner" out.txt; then
        ((win+=1))
    elif grep -q "It was a draw" out.txt; then
        ((draw+=1))
    elif grep -q "Adam was the winner" out.txt; then
        ((lose+=1))
    fi

    # Display current scores
    echo "Wins: $win"
    echo "Draws: $draw"
    echo "Losses: $lose"

    rm out.txt

    ((loopcount-=1))
done

# Display final scores
echo "FINISHED RUNNING! Final scores:"
echo "Wins: $win"
echo "Draws: $draw"
echo "Losses: $lose"
