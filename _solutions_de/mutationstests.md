---
title: Mutationstests
description: Testen der Robustheit von Software-Tests durch gezielte Codeänderungen.
category:
- Testing
problems:
- poor-test-coverage
- insufficient-testing
- regression-bugs
- legacy-code-without-tests
- quality-blind-spots
- outdated-tests
layout: solution
lang: de
en_slug: mutation-testing
related_solutions:
- slug: automated-tests
  similarity: 0.8
- slug: integration-tests
  similarity: 0.8
- slug: test-driven-development-tdd
  similarity: 0.75
- slug: security-tests
  similarity: 0.75
- slug: chaos-engineering
  similarity: 0.75
- slug: static-code-analysis
  similarity: 0.75
---

## Description

Mutationstests bewerten die Qualität einer Testsuite, statt des Codes, den sie testet, indem automatisch kleine, bewusste Änderungen — Mutanten — in den Produktionscode eingeführt werden, wie das Umdrehen einer Bedingung oder das Ändern eines arithmetischen Operators, und dann die bestehenden Tests gegen jeden Mutanten ausgeführt werden, um zu sehen, ob mindestens ein Test fehlschlägt. Ein Mutant, der keinen Test fehlschlagen lässt, ist ein „überlebender Mutant", und er zeigt konkret, dass die Testsuite diese spezifische Bugklasse nicht gefangen hätte, wäre sie natürlich aufgetreten, was den Mutationsscore zu einem weit direkteren Maß der Testeffektivität macht als Zeilen- oder Zweigabdeckung, die beide nur bestätigen, dass Code ausgeführt wurde, nicht dass sein Verhalten tatsächlich verifiziert wurde. Legacy-Systeme tragen oft ein falsches Sicherheitsgefühl rund um ihre Testsuiten: Eine Codebasis kann hohe Zeilenabdeckung zeigen, während ihre Tests nur bestätigen, dass eine Methode lief, ohne eine Exception zu werfen, ohne je zu prüfen, dass die Methode das korrekte Ergebnis produzierte, und diese Lücke ist für Abdeckungs-Tooling unsichtbar, aber genau das, was Mutationstests offenlegen. Mutationstests gegen kritische Legacy-Geschäftslogik auszuführen — statt gegen die gesamte Codebasis auf einmal, was unerträglich langsam wäre — bringt genau ans Licht, welche der bestehenden Tests tragend sind und welche dekorativ, was dem Team eine konkrete, priorisierte Liste schwacher Stellen gibt, die neu geschrieben werden sollten, bevor man sich auf diese Suite als Sicherheitsnetz für weitere Änderungen verlässt. Weil Mutationstests die vollständige Testsuite einmal pro Mutant ausführen, sind sie rechenintensiv und können auch äquivalente Mutanten erzeugen, die kein Test je erkennen könnte, weil sie das Programmverhalten tatsächlich nicht ändern, sodass sie bewusst abgegrenzt statt wahllos über eine große Legacy-Codebasis angewendet werden müssen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Führen Sie ein für die Sprache des Projekts passendes Mutationstest-Werkzeug ein (z. B. PIT für Java, Stryker für JavaScript/TypeScript)
- Beginnen Sie mit den kritischsten Geschäftslogikmodulen, statt Mutationstests über die gesamte Codebasis auszuführen
- Führen Sie Mutationstests in CI auf geänderten Dateien oder Modulen aus, um Feedback-Schleifen kurz zu halten
- Nutzen Sie den Mutationsscore als Qualitätsindikator neben Codeabdeckung, um schwache Testsuiten zu identifizieren
- Fokussieren Sie sich auf überlebende Mutanten: Jeder davon repräsentiert eine Testlücke, die einen echten Bug verbergen könnte
- Setzen Sie schrittweise Mutationsscore-Schwellenwerte, um die Testqualität über die Zeit graduell zu verbessern

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Offenbart Tests, die trotz Codeänderungen bestehen, und deckt falsches Vertrauen in die Testabdeckung auf
- Treibt die Erstellung sinnvollerer, verhaltensverifizierender Tests voran
- Identifiziert toten Code und unerreichbare Zweige, die Mutationstests nicht mutieren können
- Bietet ein genaueres Qualitätssignal als Zeilenabdeckung allein

**Kosten und Risiken:**
- Rechenintensiv: Das Ausführen Hunderter mutierter Testzyklen braucht erhebliche Zeit
- Kann äquivalente Mutanten produzieren, die unmöglich zu erkennen sind, was Rauschen erzeugt
- Kann Teams überwältigen, wenn ohne Abgrenzung auf große Legacy-Codebasen angewendet
- Erfordert bereits angemessen schnelle und stabile Testsuiten

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Banking-Anwendung hatte 85 % Zeilenabdeckung, was dem Team Vertrauen in ihre Testsuite gab. Als sie PIT-Mutationstests auf dem Kreditberechnungsmodul einführten, betrug der Mutationsscore nur 42 %, was bedeutete, dass mehr als die Hälfte der Code-Mutationen von bestehenden Tests unentdeckt blieb. Die Untersuchung offenbarte, dass viele Tests nur bestätigten, dass Methoden keine Exceptions warfen, statt korrekte Ausgabewerte zu verifizieren. Das Team schrieb die schwächsten Tests neu und erhöhte den Mutationsscore innerhalb von zwei Sprints auf 78 %, wobei dabei drei zuvor verborgene Berechnungsbugs gefangen wurden.
