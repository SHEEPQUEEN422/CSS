import time
import random

actions = ['rock', 'paper', 'scissors', 'lizard', 'spock']

def consist(x):
    act = x.lower()
    if act == 'help':
        return 'help'
    elif act in actions:
        return 'gameon'
    else:
        raise ValueError("Invalid Input:Please enter a valid action or 'help'")
        
def opponent():
    opp=random.choice(actions)
    return opp
      
def rock(play):
    if play == 'scissors' or play == 'lizard':
        return True
    return False

def paper(play):
    if play == 'rock' or play == 'spock':
        return True
    return False

def scissors(play):
    if play == 'paper' or play == 'lizard':
        return True
    return False

def lizard(play):
    if play == 'spock' or play == 'paper':
        return True
    return False

def spock(play):
    if play == 'scissors' or play == 'rock':
        return True
    return False        

myplays = dict(zip(['rock', 'paper', 'scissors', 'lizard', 'spock'], [rock, paper, scissors, lizard, spock]))

def gameon(myplays, you, comp):
    if myplays[you](comp)==True:
        return ('you',you,comp)
    elif you == comp:
        return ('tie',you,comp)
    else:
        return ('computer',you,comp)
    
def hooray(result):
    winner=result[0]
    you=result[1]
    comp=result[2]
    if winner=='you':
        return 'Hooray, you won!'
    elif winner=='tie':
        return 'It was a tie. Keep up, you got this!'
    else:
        return 'Oh no, you lost... Better luck next time!'
    
def myprompt():
    x = input("Choose a weapon or stop: >> ")
    x = x.lower()
    if x == 'stop':
        print('\n\nThanks for playing. See you next time!', end = '\n\n')
        return None
    try:
        if consist(x) == 'help':
            myhelp()
        else:
            opponent_choice = opponent()
            result = gameon(myplays, mythrill(x), opponent_choice)  
            hooraytext = f"""
You choose: {result[1].capitalize()}
The computer chose: {result[2].capitalize()}

{hooray(result)}
"""
            print(hooraytext)
    except ValueError as ve:
        print("\nThis selection is invalid. Type 'help' to check your options or 'stop' to stop the game.\n")
    myprompt()
    

def mythrill(mychoice):
    print('\n\nAlright. ', end = '')
    time.sleep(1)
    print('Now shout with me: ', end = '')
    time.sleep(1)
    print('...Rock...', end = '')
    time.sleep(0.5)
    print('...Paper...', end = '')
    time.sleep(0.5)
    print('...Scissors...', end = '')
    time.sleep(0.5)
    print('...Lizard...', end = '')
    time.sleep(0.5)
    print('...Spock!', end = '\n\n')
    time.sleep(1)
    return mychoice

def myhelp():
    myhelpstr = """
This is my Rock-Paper-Scissors-Lizard-Spock game.

You can select among the following weapons:

1. Rock: It wins against Lizard or Scissors (crushes the Lizard and breaks the Scissor)

2. Paper: Wins against Spock or Rock (disproves Spock and wraps the Rock)

3. Scissors: Wins against Lizard and Paper (kills the Lizard or cuts the Paper)

4. Lizard: Wins against Spock and Paper (poisons Spock or eats the Paper)

5. Spock: Wins against Rock and Paper (vaporizes the Rock and smashes the Scissors)

To choose a weapon, just type its name. No need to type in lower or upper case. The prompt is smart 
enough to choose the right action here.

When you are done, you can leave the game by typing 'stop'.

Game on!
"""
    print('\n\n')
    print(myhelpstr, end = '\n\n')
    
myprompt()
