var app = new Vue({
  el: '#app',
  data: {
    word: '',
    cbow: ['-'],
    skipgram: ['-']
  },
  methods: {
    search(e) {
      e.preventDefault()
      fetch('http://localhost:5000/get-similar', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({word: this.word})
      })
      .then(res => res.json())
      .then(res => {
        if (res.cbow && res.skipgram) {
          this.cbow = res.cbow
          this.skipgram = res.skipgram	
        } else {
          this.cbow = ['Невідоме слово']
          this.skipgram = ['Невідоме слово']
        }

        this.word = res.word
      })
    }
  }
})