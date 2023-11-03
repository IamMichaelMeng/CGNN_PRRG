from Foursquare_recommendation import main as Foursquare_main
from Gowalla_recommendation import main as Gowalla_main
from Yelp_recommendation import main as Yelp_main

if __name__ == '__main__':
    num_group = 60  # 群组的数量
    GROUP_SIZE = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    epoch = 16
    # Foursquare_main(num_group, GROUP_SIZE, epoch)
    Gowalla_main(num_group, GROUP_SIZE, epoch)
    Yelp_main(num_group, GROUP_SIZE, epoch)
