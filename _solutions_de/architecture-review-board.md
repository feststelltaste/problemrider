---
title: Architecture Review Board
description: Einrichtung eines Gremiums zur Überwachung und Steuerung der Architekturentwicklung.
category:
- Architecture
- Management
problems:
- stagnant-architecture
- technology-stack-fragmentation
- inconsistent-codebase
- architectural-mismatch
- decision-avoidance
- convenience-driven-development
- high-technical-debt
- delayed-decision-making
- project-authority-vacuum
layout: solution
lang: de
en_slug: architecture-review-board
related_solutions:
- slug: architecture-reviews
  similarity: 0.8
- slug: architecture-governance
  similarity: 0.8
- slug: architecture-decision-records
  similarity: 0.8
- slug: architecture-roadmap
  similarity: 0.7
- slug: architecture-documentation
  similarity: 0.7
- slug: architecture-conformity-analysis
  similarity: 0.7
---

## Description

Ein Architecture Review Board ist ein ständiges Gremium, zusammengesetzt aus Senior-Architekten und Vertretern jedes größeren Entwicklungsteams, beauftragt, bedeutende architektonische Entscheidungen — teamübergreifende Änderungen, neue Technologieeinführungen, größere Refaktorierungen — in regelmäßigem Turnus zu überprüfen und zu genehmigen, statt nur in einer Krise zusammenzukommen. In Legacy-Umgebungen mit mehreren Teams, die unabhängig voneinander eine gemeinsame Plattform pflegen, bedeutet die Abwesenheit eines solchen Gremiums, dass architektonische Entscheidungen isoliert Team für Team getroffen werden, was dazu führen kann, dass eine gemeinsame Codebasis beispielsweise mehrere verschiedene ORMs oder Authentifizierungsmechanismen aufweist, jeweils unabhängig eingeführt, um dasselbe zugrunde liegende Problem auf leicht unterschiedliche Weise zu lösen. Das Board adressiert dies, indem es ein einziges Forum bietet, in dem konkurrierende Technologieentscheidungen gemeinsam bewertet werden und eine Entscheidung einmal, im Namen der gesamten Organisation, getroffen wird, wobei diese Entscheidung zusammen mit ihrer Begründung — und abweichenden Meinungen — veröffentlicht wird, sodass Teams nicht nur verstehen, was entschieden wurde, sondern warum. Weil die Entscheidungen des Boards organisatorisches Gewicht tragen, ist es auch der natürliche Ort zur Koordination von Modernisierungsarbeit über Teams hinweg, deren Änderungen architektonisch miteinander und mit einem gemeinsamen Zielzustand kompatibel bleiben müssen — etwas, das kein einzelnes Team allein durchsetzen kann. Das Board klein zu halten, ihm eine enge Charta zu geben, die genau definiert, welche Entscheidungen seine Überprüfung erfordern, und es in einem kurzen, regelmäßigen Turnus tagen zu lassen, verhindert, dass es entweder zu einer Abstempel-Formalität oder einem Engpass wird, der die Lieferung über die Organisation hinweg verlangsamt. Das Hauptrisiko bleibt, dass Board-Mitglieder, die von der täglichen Entwicklung abgekoppelt sind, Entscheidungen genehmigen könnten, die architektonisch fundiert, aber praktisch undurchführbar sind, oder dass das Board zu reflexartigem Konservatismus abdriftet, der notwendige Änderungen blockiert, weil sie kurzfristiges Risiko einführen.

## How to Apply ◆

> In Legacy-Umgebungen bietet ein Architecture Review Board die organisatorische Struktur, die nötig ist, um bewusste, koordinierte architektonische Entscheidungen zu treffen, statt das System durch unkoordinierte Einzelentscheidungen weiter verfallen zu lassen.

- Bilden Sie ein Board mit Vertretern jedes größeren Entwicklungsteams plus Senior-Architekten, und halten Sie es klein genug, um Entscheidungen effizient zu treffen (fünf bis acht Mitglieder funktionieren typischerweise gut).
- Definieren Sie eine klare Charta, die festlegt, welche Entscheidungen Board-Überprüfung erfordern (teamübergreifende Änderungen, neue Technologieeinführungen, größere Refaktorierungen) und welche an einzelne Teams delegiert werden.
- Tagen Sie regelmäßig in kurzem Turnus (alle zwei Wochen oder monatlich) mit einer strukturierten Agenda, statt nur für größere Entscheidungen zusammenzukommen, sodass das Board über die laufende architektonische Entwicklung informiert bleibt.
- Veröffentlichen Sie alle Board-Entscheidungen, einschließlich Begründung und abweichender Meinungen, in einem zugänglichen Entscheidungsprotokoll, sodass Teams nicht nur verstehen, was entschieden wurde, sondern warum.
- Nutzen Sie das Board zur Koordination von Modernisierungsbemühungen über Teams hinweg und stellen Sie sicher, dass die Änderungen verschiedener Teams architektonisch kompatibel sind und sich auf einen gemeinsamen Zielzustand zubewegen.
- Überprüfen Sie die Effektivität des Boards periodisch und passen Sie seinen Umfang und seine Prozesse an, um zu verhindern, dass es entweder zu einer Abstempel-Formalität oder einem Engpass wird.

## Tradeoffs ⇄

> Ein Architecture Review Board bietet koordinierte architektonische Richtung, kann aber zu einem Engpass oder Elfenbeinturm werden, wenn es nicht sorgfältig gemanagt wird.

**Vorteile:**

- Verhindert unkoordinierte Technologiewucherung, indem ein Forum zur Bewertung und Genehmigung neuer Technologieeinführungen bereitgestellt wird.
- Stellt teamübergreifende architektonische Konsistenz sicher, was besonders wichtig ist, wenn mehrere Teams verschiedene Teile desselben Legacy-Systems ändern.
- Schafft Verantwortlichkeit für architektonische Entscheidungen und verringert die Tendenz, schwierige Entscheidungen unbegrenzt aufzuschieben.
- Bietet einen Ort zum Teilen architektonischen Wissens und Musters über Teams hinweg, die sonst isoliert arbeiten könnten.

**Kosten und Risiken:**

- Ein Board, das für zu viele Entscheidungen Genehmigung verlangt, wird zu einem Engpass, der die Entwicklung verlangsamt und Teams frustriert.
- Board-Mitglieder, die von der täglichen Entwicklung abgekoppelt sind, könnten Entscheidungen treffen, die theoretisch fundiert, aber praktisch undurchführbar sind.
- Ohne klare Delegationsregeln könnten Teams unsicher sein, ob sie Board-Genehmigung benötigen, was entweder zu unnötigen Verzögerungen oder ungenehmigten Änderungen führt.
- Das Board könnte eine Neigung zu Konservatismus entwickeln und notwendige Änderungen abwehren, weil sie kurzfristiges Risiko einführen.

## How It Could Be

> Das folgende Szenario zeigt, wie ein Architecture Review Board die Legacy-Modernisierung über Teams hinweg koordiniert.

Eine Regierungsbehörde mit sechs Entwicklungsteams, die eine gemeinsame Legacy-Plattform pflegten, richtete ein Architecture Review Board ein, nachdem entdeckt wurde, dass drei Teams unabhängig voneinander begonnen hatten, verschiedene Microservices-Frameworks einzuführen. Das Board bewertete alle drei Optionen, wählte eine als Standard und erstellte Migrationsrichtlinien, denen alle Teams folgen würden. Sie etablierten außerdem einen „Technologie-Radar", der Technologien in vier Kategorien einteilte: übernehmen, testen, bewerten und zurückhalten. Der Radar machte klar, welche Technologien für den Produktionseinsatz genehmigt waren und welche noch bewertet wurden. Über zwei Jahre überprüfte das Board 45 bedeutende architektonische Vorschläge, genehmigte 38 (oft mit Änderungen) und lehnte 7 mit Erklärungen ab. Die abgelehnten Vorschläge beinhalteten zwei Fälle, in denen Teams Technologien einführen wollten, die bereits auf der „Zurückhalten"-Liste standen, was weitere Fragmentierung verhinderte.
