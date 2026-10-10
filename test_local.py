import unittest
import sys
import types
# Offline test runner can work without the API SDK installed.
try:
    from openai import OpenAI
except (ImportError, AttributeError):
    stub = types.ModuleType("openai")
    stub.OpenAI = object
    sys.modules["openai"] = stub
from benchmark import (Sample, answer_messages, choose_target, finance_correct,
                       indices_from_output, score, text_metrics, table_to_text)
import random
import tempfile
from pathlib import Path
from benchmark import paired_ci, summarize

class OfflineTest(unittest.TestCase):
    def setUp(self):
        self.s = Sample('x','easy_coqa','The apple is red.',
                        [{'q':'What color is the apple?','a':'red'},
                         {'q':'What fruit is it?','a':'apple'}],
                        'Was it red?', 'yes')

    def test_prompts(self):
        raw = answer_messages(self.s,'raw_multiturn',None,None)
        select = answer_messages(self.s,'history_selection',[1],None)
        rewrite = answer_messages(self.s,'one_line_rewrite',None,'Was the apple red?')
        self.assertEqual([m['role'] for m in raw], ['system','user','assistant','user','assistant','user'])
        self.assertEqual(len(select), 4)
        self.assertEqual(len(rewrite), 2)
        self.assertIn('SOURCE DOCUMENT', rewrite[0]['content'])
        self.assertNotIn('What color is', rewrite[-1]['content'])

    def test_indices(self):
        self.assertEqual(indices_from_output('[2, 0]',3),[0,2])
        with self.assertRaises(ValueError): indices_from_output('[0,0]',3)
        with self.assertRaises(ValueError): indices_from_output('probably [1]',3)

    def test_text(self):
        self.assertEqual(text_metrics('The red apple!', 'red apple'),(1,1))
        self.assertEqual(score(self.s,'yes')['correct'],1.0)

    def test_finance(self):
        self.assertEqual(finance_correct('12%', 0.12), 1.0)
        self.assertEqual(finance_correct('12%', 12.0), 1.0)
        self.assertEqual(finance_correct('USD 1000', 1000), 1.0)
        self.assertEqual(finance_correct('I cannot tell', 0.12), 0.0)
        self.assertEqual(finance_correct('Yes','yes'),1.0)

    def test_target(self):
        idx = choose_target(['a','b','c','d','e'],['1','2','3','4','5'],random.Random(1),3,4)
        self.assertIn(idx,[3,4])

    def test_table(self):
        self.assertIn('Revenue: 3', table_to_text({'2019':{'Revenue':3}}))

    def test_bootstrap_and_summary(self):
        records = []
        for suite in ('easy_coqa','hard_convfinqa'):
            for variant, cor in [('raw_multiturn', 0),
                                 ('history_selection', 1),
                                 ('one_line_rewrite', 0)]:
                records.append({'id':suite+'1','suite':suite,'variant':variant,
                                'question':'Test?', 'gold':'yes', 'correct':cor,
                                'f1':cor, 'error':None})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'summary.md'
            summarize(records,path,1)
            body = path.read_text()
            self.assertIn('Paired wins / losses', body)
            self.assertIn('Selected disagreements', body)
            self.assertIn('+1.000', body)
        self.assertEqual(paired_ci([(records[1],records[0])])[0],1.0)




# Additional tests are declared below the original __main__ block so that
# unittest discovery (python -m unittest) can execute them.
class ResponseRepairTests(unittest.TestCase):
    def setUp(self):
        import benchmark
        self.bench = benchmark
        self.original = getattr(benchmark._client_tls, "client", None)

    def tearDown(self):
        if self.original is None:
            if hasattr(self.bench._client_tls, "client"):
                del self.bench._client_tls.client
        else:
            self.bench._client_tls.client = self.original

    def _reply(self, text, status="completed", reason=None, out_tokens=10, reasoning_tokens=2):
        return types.SimpleNamespace(
            output_text=text, status=status,
            incomplete_details=types.SimpleNamespace(reason=reason) if reason else None,
            usage=types.SimpleNamespace(input_tokens=12, output_tokens=out_tokens,
                                        output_tokens_details=types.SimpleNamespace(
                                            reasoning_tokens=reasoning_tokens)))

    def _fake_client(self, responses):
        class API:
            def __init__(self, arr):
                self.arr = iter(arr)
                self.kwargs = []
            def create(self, **kwargs):
                self.kwargs.append(kwargs)
                response = next(self.arr)
                if isinstance(response, Exception):
                    raise response
                return response
        api = API(responses)
        self.bench._client_tls.client = types.SimpleNamespace(responses=api)
        return api

    def test_empty_reasoning_then_structured_answer(self):
        api = self._fake_client([
            self._reply("", status="incomplete", reason="max_output_tokens",
                        out_tokens=1800, reasoning_tokens=1800),
            self._reply('{"indices":[0,2]}', out_tokens=25)
        ])
        sample = Sample('id', 'easy_coqa', 'example',
                        [{'q': 'a', 'a': '1'}, {'q': 'b', 'a': '2'}, {'q': 'c', 'a': '3'}],
                        'What was it?', '1')
        chosen, r = self.bench.select_history(sample, 'gpt-6-luna', 'low')
        self.assertEqual(chosen, [0, 2])
        self.assertEqual(r['api_attempts'], 2)
        self.assertEqual(r['output_tokens'], 1825)
        self.assertEqual([x['max_output_tokens'] for x in api.kwargs], [1800, 3600])
        self.assertEqual(api.kwargs[0]['text']['format']['type'], 'json_schema')

    def test_bad_json_then_valid_legacy_array(self):
        self._fake_client([self._reply('bad result'), self._reply('[1]')])
        result = self.bench.llm('gpt-6-luna', [{'role':'user','content':'JSON'}], 'low',
                                max_output_tokens=512,
                                validator=lambda text: self.bench.parse_selector_output(text, 3))
        self.assertEqual(result['parsed'], [1])
        self.assertEqual(result['api_attempts'], 2)

    def test_unusable_response_diagnostic(self):
        self._fake_client([self._reply('', status='incomplete', reason='max_output_tokens',
                                      out_tokens=512, reasoning_tokens=512)])
        with self.assertRaisesRegex(ValueError, 'reasoning_tokens=512'):
            self.bench.llm('gpt-6-luna', [{'role':'user', 'content':'Hi'}],
                           'low', 512, retries=0)

    def test_compact_recovered_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.jsonl'
            for row in [{'id': 'x', 'variant': 'history_selection', 'error': 'oops'},
                        {'id': 'x', 'variant': 'history_selection', 'error': None}]:
                self.bench.append_jsonl(path, row)
            self.bench.compact_jsonl(path)
            rows = self.bench.read_jsonl(path)
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]['error'])


class SummaryIntegrityTests(unittest.TestCase):
    def test_history_counts_and_failure_aware_pairing(self):
        # A failure should reduce operational accuracy but must not be treated
        # as an algorithmic accuracy loss in the both-completed comparison.
        rows = []
        for suite in ('easy_coqa','hard_convfinqa'):
            for idx in range(2):
                for variant in ('raw_multiturn','history_selection','one_line_rewrite'):
                    failed = idx == 0 and variant == 'history_selection'
                    rows.append({'id':f'{suite}:{idx}', 'suite':suite, 'variant':variant,
                                 'history_n':4, 'kept_n':2 if variant == 'history_selection' else None,
                                 'correct':float(not failed), 'f1':float(not failed),
                                 'error':'mock failure' if failed else None,
                                 'prediction':'' if failed else 'yes',
                                 'question':'What?', 'gold':'yes'})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'summary.md'
            summarize(rows, path, 2)
            md = path.read_text()
            self.assertIn('Mean kept turns | Errors | Completed accuracy',md)
            self.assertIn('Raw', 'Raw')
            self.assertIn('| easy_coqa | raw_multiturn | 2/2 | 1.000 | 1.000',md)
            self.assertIn('| 4.00 | 0 | 1.000 |',md)
            self.assertIn('| easy_coqa | history_selection | 2/2 | 0.500 | 0.500',md)
            self.assertIn('| 2.00 | 1 | 1.000 |',md)
            self.assertIn('| easy_coqa | history_selection vs raw_multiturn | 1 | 0 | 1 | 0 | +0.000 |',md)
            self.assertIn('up to 2 examples per dataset',md)

if __name__ == '__main__': unittest.main()
