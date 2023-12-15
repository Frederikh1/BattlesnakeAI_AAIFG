@echo off
set win=0 & set draw=0 & set lose=0 & set loopcount=100
:: Adam url: https://battlesnake.up.railway.app/
:: Replit url: https://battlesnakeaiaaifg.kasperdelaxson.repl.co/
:: localhost url: http://localhost:8000/
:loop
::"../battlesnake/battlesnake.exe" play -W 11 -H 11 --name AFK --url https://battlesnakeaiaaifg.kasperdelaxson.repl.co/ --name Adam --url https://battlesnake.up.railway.app/ 1> out.txt 2>&1
::uncomment for local test:
"../battlesnake/battlesnake.exe" play -W 11 -H 11 --name AFK --url http://localhost:8000 --name Adam --url https://battlesnake.up.railway.app/ 1> out.txt 2>&1
findstr /c:"AFK was the winner" "out.txt" >nul 2>&1 && set /a win=win+1
findstr /c:"It was a draw" "out.txt" >nul 2>&1 && set /a draw=draw+1
findstr /c:"Adam was the winner" "out.txt" >nul 2>&1 && set /a lose=lose+1
ECHO Wins: %win% & echo:Draws: %draw% & echo:Losses: %lose%
del out.txt

set /a loopcount-=1
if %loopcount% gtr 0 goto loop
:exitloop

ECHO FINISHED RUNNING! Final scores: & echo:Wins: %win% & echo:Draws: %draw% & echo:Losses: %lose%
pause
