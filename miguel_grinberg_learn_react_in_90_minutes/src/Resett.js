import React from 'react';

const Resett = React.memo(({setCounter}) => {
    console.log('Resett')
    const reset = () => {setCounter(0);}

    return (
        <button onClick={reset}>Reset</button>
    );
});

export default Resett;