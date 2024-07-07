let idleTime = 0;

window.onload = function() {
    // Increment the idle time counter every minute.
    let idleInterval = setInterval(timerIncrement, 60000); // 1 minute

    // Zero the idle timer on mouse movement.
    window.onmousemove = resetTimer;
    window.onkeydown = resetTimer;
    window.onscroll = resetTimer;
    window.onclick = resetTimer;
};

function timerIncrement() {
    idleTime += 1;
    if (idleTime >= 1) { // 15 minutes
        
        window.location.href = '/logout';
        alert("Session expired due to inactivity.");
    }
}

function resetTimer() {
    idleTime = 0;
}
