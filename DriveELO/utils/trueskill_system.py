from trueskill import Rating, TrueSkill
from trueskill import rate_1vs1

def main():
    env = TrueSkill(mu=1600, sigma=350)
    env.make_as_global()

    player_A_rating = Rating(mu=1500, sigma=350)
    player_B_rating = Rating(mu=1500, sigma=350)

    for i in range(100):
        player_A_rating, player_B_rating = rate_1vs1(player_A_rating, player_B_rating)
        print(f"After round {i+1}:")
        print(f"New Rating for Player A: {player_A_rating}")
        print(f"New Rating for Player B: {player_B_rating}")

if __name__ == "__main__":
    main()
