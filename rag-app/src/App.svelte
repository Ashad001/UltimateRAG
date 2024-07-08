<script>
	import FeedbackList from "./components/FeedbackList.svelte";
	let feedback = [
		{
			id: 1,
			rating: 5,
			text: 'Great app!'
		},
		{
			id: 2,
			rating: 3,
			text: 'Could be better'
		},
		{
			id: 3,
			rating: 4,
			text: 'Awesome!'
		}
	]

</script>

<main class="container">
	<FeedbackList {feedback}/>
</main>
