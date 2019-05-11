let fake = [{"url": "nimasil.com8", "score": 0.8},
            {"url": "nimasil.com7", "score": 0.7},
            {"url": "nimasil.com4", "score": 0.4},
            {"url": "nimasil.com6", "score": 0.6},
            {"url": "nimasil.com6", "score": 0.6},
            {"url": "nimasil.com8", "score": 0.8}];

let res = [];

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function get_result() {
  let topic = document.getElementById('topic').value;
  let url = document.getElementById('url').value;
  let page = document.getElementById('page').value;
  if (topic.length === 0 || topic.length === 0 || topic.length === 0 ) {
    alert("Please type all three parameters into the input box.");
    return;
  }

  document.getElementById('main').style.display = 'none';
  document.getElementById('result').style.display = 'block';

  document.getElementById('info_topic').innerHTML = topic;
  document.getElementById('info_url').innerHTML = url;
  document.getElementById('info_page').innerHTML = page;

  request_start(topic, url, page);

  while (res.length < page) {
    document.getElementById('info_res1').innerHTML = '';
    document.getElementById('info_res2').innerHTML = '';
    request_update(topic, url, page);
    await sleep(2000);
  }

}

function request_start(topic, url, page) {
  $.ajax({
    type: "POST",
    url: "http://127.0.0.1:5000/start_",
    data: {topic: topic,
           url: url,
           page: page}
  }).done(function(e) {
    console.log(e);
  });
}

function request_update(topic, url, page) {
  $.ajax({
    type: "POST",
    url: "http://127.0.0.1:5000/update_",
    data: {topic: topic,
           url: url,
           page: page}
  }).done(function(e) {
    res = e;
    res.sort(function(a, b){
      return b.score - a.score;
    });

    for (let i = 0; i < res.length; i++) {
      let line = document.createElement("a");
      line.style = "color: #0084FF; display: block; border: solid 1px #D9D9D9; margin: 2vw; padding: 1vw; border-radius: 1vw; overflow: auto;";
      line.innerHTML = res[i].url;
      let score = document.createElement("a");
      score.style = "color: #0084FF; display: block; border: solid 1px #D9D9D9; margin: 2vw; padding: 1vw; border-radius: 1vw";
      score.innerHTML = Math.round(res[i].score * 1000) / 1000;
      document.getElementById('info_res1').appendChild(line);
      document.getElementById('info_res2').appendChild(score);
    }
  });
}

function go_back_home() {
  document.getElementById('main').style.display = 'block';
  document.getElementById('result').style.display = 'none';
}
