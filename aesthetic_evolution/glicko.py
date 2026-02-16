

import math
from typing import List

Q = math.log(10) / 400
MIN_DEVIATION = 30.0

class Player:
    """
    Player class representing a player in the Glicko rating system, 
    which is used to calculate player ratings based on match outcomes.
    @author: Stephen Krol
    @date: Feb 2026
    """

    def __init__(self, 
                 id: int,
                 rating: float = 1500.0,
                 deviation: float = 350.0) -> None:
        """
        Constructor for Player class representing a player in the Glicko rating system.
        @author: Stephen Krol
        @date: Feb 2026

        :param self: The instance of the Player class being created.
        :type self: Player
        :param id: The ID of the player.
        :type id: int
        :param rating: The initial rating of the player (default is 1500.0).
        :type rating: float
        :param deviation: The initial rating deviation of the player (default is 350.0).
        :type deviation: float

        :return: None
        :rtype: None
        """

        self.id = id
        self.rating = rating
        self.deviation = max(deviation, MIN_DEVIATION)

    def update_rating(self, opponent_ratings: List[float], opponent_deviations: List[float], outcomes: List[int], c:float=15.0) -> None:
        """
        Update the player's rating and deviation based on the results of matches against opponents.
        @author: Stephen Krol
        @date: Feb 2026

        :param self: The instance of the Player class being updated.
        :type self: Player
        :param opponent_ratings: A list of ratings of the opponents.
        :type opponent_ratings: List[float]
        :param opponent_deviations: A list of rating deviations of the opponents.
        :type opponent_deviations: List[float]
        :param outcomes: A list of match outcomes (1 for win, 0 for loss, 0.5 for draw).
        :type outcomes: List[int]
        :param c: The constant representing the rate of increase in rating deviation over time (default is 15.0).
        :type c: float

        :return: None
        :rtype: None
        """

        current_deviation = self.deviation
        self._pre_rating_period_update(c=c)

        d_squared_inv = 0.0
        rating_diff_sum = 0.0
        
        for opponent_rating, opponent_deviation, outcome in zip(opponent_ratings, opponent_deviations, outcomes):
            g_opponent = self._g(opponent_deviation)
            E_opponent = self._E(opponent_rating, opponent_deviation)
            
            d_squared_inv += (g_opponent**2) * E_opponent * (1 - E_opponent)
            rating_diff_sum += g_opponent * (outcome - E_opponent)
        
        d_squared_inv *= Q**2

        if d_squared_inv > 0:
            self.rating += (Q / ((1 / current_deviation**2) + d_squared_inv)) * rating_diff_sum
            self.deviation = max(math.sqrt(1 / ((1 / current_deviation**2) + d_squared_inv)), MIN_DEVIATION)

    def _pre_rating_period_update(self, c: float = 0) -> None:
        """
        Update the player's rating deviation before a new rating period begins, accounting for time decay.
        @author: Stephen Krol
        @date: Feb 2026

        :param self: The instance of the Player class being updated.
        :type self: Player
        :param c: The constant representing the rate of increase in rating deviation over time (default
        is 0, meaning no time decay).
        :type c: float

        :return: None
        :rtype: None
        """

        self.deviation = min(math.sqrt(self.deviation**2 + c**2), 350.0)
    
    def _g(self, opponent_deviation: float) -> float:
        """
        Calculate the g() function used in the Glicko rating system, which adjusts 
        the impact of an opponent's rating deviation.
        @author: Stephen Krol
        @date: Feb 2026

        :param self: The instance of the Player class.
        :type self: Player
        :param opponent_deviation: The rating deviation of the opponent.
        :type opponent_deviation: float

        :return: The value of the g() function for the given opponent's rating deviation.
        :rtype: float
        """

        return 1 / math.sqrt(1 + (3 * Q**2 * opponent_deviation**2) / math.pi**2)
    
    def _E(self, opponent_rating: float, opponent_deviation: float) -> float:
        """
        Calculate the E() function used in the Glicko rating system,
        which represents the expected score against an opponent.
        @author: Stephen Krol
        @date: Feb 2026

        :param self: The instance of the Player class.
        :type self: Player
        :param opponent_rating: The rating of the opponent.
        :type opponent_rating: float
        :param opponent_deviation: The rating deviation of the opponent.
        :type opponent_deviation: float

        :return: The expected score against the given opponent.
        :rtype: float
        """

        return 1 / (1 + 10**(-self._g(opponent_deviation) * (self.rating - opponent_rating) / 400))