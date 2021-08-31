import React from 'react';
import logo from './logo.svg';
import './App.css';
import Child from './Child.js'
import Resett from './Resett.js'

function App() {
  const username = 'nadi';
  const kids = ['carmel', 'shaked', 'yaari']
  const [counter, setCounter] = React.useState(0);
  // const rates = {'G&P': 1.22, 'EUR': 0.90,}
  const [rates, setRates] = React.useState({});
  
  
  React.useEffect(() => {
    fetch('https://api.ratesapi.io/api/latest?base=USD').then(res => res.json()).then(data => {
      setRates(data.rates);
    });
  }, []);

  const text = React.useRef();
  const onFocus = () => {text.current.style.background = '#ddf';}
  const onBlur = () => {text.current.style.background = '#fff';}

  React.useEffect(() => {
    console.log(text.current)
    const myText = text.current;
    // text.current.style.background = '#ddf'
    text.current.addEventListener('focus', onFocus);
    text.current.addEventListener('blur', onBlur)

    return () => {
      myText.removeEventListener('focus', onFocus)
      myText.removeEventListener('blur', onBlur)
    }
  }, []);
  
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        {username ?
        <p>Hello {username}</p>
        :
        <p>Hello stranger</p>
        }
        <a className="App-link" href="https://reactjs.org" target="_blank" rel="noopener noreferrer">
          Learn React
        </a>
        {username && <p> user login </p>}
        <ul>
          {kids.map(kids => <li>{kids}</li>)}
        </ul>
        <p>Counter: {counter}</p>
        <Child setCounter={setCounter} n={1}/>
        <Child setCounter={setCounter} n={10}/>
        
        <h2>USD Exchange Rates</h2>
        {Object.keys(rates).map(currency => <li>{currency}: {rates[currency]}</li>)}
        <input type="text" ref={text}/> 

        <Resett setCounter={setCounter}/>
      </header>
    </div>
  );
}

export default App;
