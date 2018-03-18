#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""A simple client demonstrating how to send requests to the TensorFlask
Sentiment classification webservice.

Arguments:
    --server=(string):URL to webserver
                      [default value="http://127.0.0.1:8000/"]

Example:
    $python3 test_client.py 
    --server="http://127.0.0.1:8000/"
"""
import os
import argparse
import requests
import base64
import glob
import numpy as np


def main():
    #Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--server",
                        default='http://127.0.0.1:8000/',
                        help="URL to webserver",
                        type=str)
    args = parser.parse_args()
    requests_list = [
        {"sentence" : "I'm positive she didn't notice us the first time."},
        {"sentence" : "Positive. She smelled amazingly human, only more so."},
        {"sentence" : "Janetta doesn’t miss her mom"},
        {"sentence" : "Everybody gets the day off."},
        {"sentence" : " Nobody gets the day off"},
        {"sentence" : "When you see good actors in a project like this, you wonder if they signed up as an alternative to canyoneering"},
        {"sentence" : "An empty-headed horror movie with nothing to recommend it"},
        {"sentence" : "Star Wars is a junkyard of cinematic gimcracks not unlike the Jawas' heap of purloined, discarded, barely functioning droids"},
        {"sentence" : "A theme park ride masquerading as master's thesis"},
        {"sentence" : "Plodding, puffed-up kitsch mistaking itself for profound psycho noir that the source material won't support"},
        {"sentence" : 'Among the slackest, laziest, least movie-like movies released by a major studio in the last decade, "Grown Ups 2" is perhaps the closest Hollywood has yet come to making "Ow! My Balls!" seem like a plausible future project.'},
        {"sentence" : 'Two chemists walk into a bar. The first one says “I’ll have H2O”. The second one says “I’ll have H2O too”. The second one dies.'}
    ]

    #Send request to server
    print("Requesting classifications for {} sentences...".format(len(requests_list)))
    response = requests.post(args.server+'/sentiment/classify', json={'requests' : requests_list})

    #Parse JSON response from server
    json_response = response.json()
    print('JSON response from server :')
    print('    ',json_response,'\n')
    if(response.status_code == 200):
        print('Classifications returned from server :')
        for sentence, response in zip(requests_list, json_response['responses']):
            print('    sentence {} classified as a {} with approximate probability/score {}'
                  .format(sentence["sentence"], response['class'], response['probability']))

if __name__ == '__main__':
    main()  
