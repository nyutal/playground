import React from 'react';

export default function Child({setCounter, n}) {
    const inc = () => { setCounter(x => x + n);};
    const dec = () => { setCounter(x => x - n);};

    return  (
        <div>
          <button onClick={inc}> +{n} </button>
          <button onClick={dec}> -{n} </button>
        </div>
    );
}