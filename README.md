## KerasFlask
A simple web service classifying sentiment of sentences from HTTP POST requests built using [Flask](http://flask.pocoo.org/), [Keras](https://keras.io/) and training on [Twitter data](http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip)

The API uses HTTP POST operations to classify sentences that is sent in the request. The single POST request available is /sentiment/classify.
The API uses JSON for both requests and responses, see below for a detailed specification of the JSON data format.


## JSON request format
The HTTP POST request /sentiment/classify expects a JSON request. Example JSON data for the request:
```python
{
  "requests":[
      {
        "sentence":"You would think that taking off a snail’s shell would make it move faster, but it actually just makes it more sluggish."
      },
      {
        "sentence":"Two chemists walk into a bar. The first one says “I’ll have H2O”. The second one says “I’ll have H2O too”. The second one dies."
      }
  ]
}
```
* requests - A list of requests, one for each sentence
    * sentence - A english sentence for which the sentiment to be classified

## JSON response format
```python
{
  "responses":[
      {
        "class":positive,
        "probability":0.98,
        "sentence":"You would think that taking off a snail’s shell would make it move faster, but it actually just makes it more sluggish."
      },
      {
        "class":positive,
        "probability":0.6028993129730225,
        "sentence":""Two chemists walk into a bar. The first one says “I’ll have H2O”. The second one says “I’ll have H2O too”. The second one dies.""
      }
  ]
}
```
* responses - A list of responses, one for each sentence
    * class - sentiment of sentence [positive,negative]
    * probability - The inferred probability of the predicted sentiment [positive,negative] (Softmax score)

## Installing requirements
The API uses python3 and the requirements can be installed by

```bash
    $pip3 install -r requirements.txt
```

## Running the server
The Flask application can be deployed using e.g. gunicorn using:

```bash
    $gunicorn server_application:app
```

## Running the simple test client
After starting the server requests can be sent using the test client.
For detailed use of the test client see:

```bash
    $python3 simple_client.py --server="http://127.0.0.1:8000/
```



## Training new model parameters

```bash
    $python3 python3 train_model.py
```
