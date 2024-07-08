<script>
    import Card from "./card.svelte";

    export let item = {id: 0, rating: 0, text: ''};


</script>

<Card>
<div class="num-display">
    <p>{item.rating}</p>
</div>
<button class="close">X</button>
<div class="feedback">
    <p>{item.text}</p>
</div>
</Card>

<style>
 .num-display {
    position: absolute;
    top: -10px;
    left: -10px;
    width: 50px;
    height: 50px;
    background: #ff6e95;
    color: #fff;
    border: 1px #eee solid;
    border-radius: 50%;
    padding: 10px;
    text-align: center;
    font-size: 19px;
 }

 .close {
    position: absolute;
    top: 10px;
    right: 20px;
    cursor: pointer;
}

</style>