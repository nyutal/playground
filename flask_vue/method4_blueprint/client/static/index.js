const apiEndpoint = '/api_v1/';

const vm = new Vue({ // Again, vm is our Vue instance's name for consistency.
    el: '#vm',
    delimiters: ['[[', ']]'],
    data: {
       greeting: 'Hello, Vue!',
        flaskGreeting: ''
    },
    created: async function(){
        const gResponse = await fetch(apiEndpoint + 'greeting');
        const gObject = await gResponse.json();
        this.flaskGreeting = gObject.greeting;
    }
})
