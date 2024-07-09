<script>
    export let count = 0;
    export let average = 0.0;


</script>

<div class="feedback-stats">
    <h4> {count} Reviews</h4>
    <h4> Average Rating: {average.toFixed(1)} </h4>
</div>

<style>
    .feedback-stats {
        display: flex;
        justify-content: space-between;
        margin: 20px 0;
    }
</style>