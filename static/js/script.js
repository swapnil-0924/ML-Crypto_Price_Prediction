// Sidebar Toggle
$(document).ready(function () {
    // Toggle sidebar
    $('#sidebarCollapse').on('click', function () {
        $('#sidebar').toggleClass('active');
        $('#content').toggleClass('active');
    });

    // Auto-close sidebar on mobile
    $(window).on('resize', function () {
        if ($(window).width() <= 768) {
            $('#sidebar').addClass('active');
            $('#content').addClass('active');
        }
    });

    // Smooth scroll for charts
    $('a[href^="#"]').on('click', function (e) {
        e.preventDefault();
        const target = $(this).attr('href');
        $('html, body').animate({
            scrollTop: $(target).offset().top - 100
        }, 800); // 800ms scroll duration
    });

    // Add hover effects to metric boxes
    $('.metric-box').hover(
        function () {
            $(this).css('transform', 'scale(1.05)');
        },
        function () {
            $(this).css('transform', 'scale(1)');
        }
    );

    // Form Validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function (e) {
            const inputs = this.querySelectorAll('input, textarea');
            let isValid = true;

            inputs.forEach(input => {
                if (!input.checkValidity()) {
                    isValid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });

            if (!isValid) e.preventDefault();
        });
    });

    // Particles Background Initialization
    particlesJS('particles-canvas', {
        particles: {
            number: { value: 80 },
            color: { value: '#6366f1' },
            shape: { type: 'circle' },
            opacity: { value: 0.5 },
            size: { value: 3 },
            move: { enable: true, speed: 1 }
        },
        interactivity: {
            events: {
                onhover: { enable: true, mode: 'repulse' }
            }
        }
    });
});