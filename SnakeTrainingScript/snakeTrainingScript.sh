#!/bin/bash
win=0
draw=0
lose=0
loopcount=2

# Loop through the game plays
while [ $loopcount -gt 0 ]; do
    ../battlesnake/battlesnake.exe play -W 11 -H 11 --name AFK --url https://battlesnakeaiaaifg.kasperdelaxson.repl.co/ --name Adam --url https://battlesnake.up.railway.app/ 1> out.txt 2>&1

    # Checking game outcomes and updating counters
    grep -q "AFK was the winner" out.txt && ((win+=1))
    grep -q "It was a draw" out.txt && ((draw+=1))
    grep -q "Adam was the winner" out.txt && ((lose+=1))

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